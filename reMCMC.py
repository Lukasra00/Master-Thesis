import numpy as np
import ipdb
import logging
import os
import wandb
from copy import deepcopy
import pynvml
import time 
import threading
import ray

from argparse import ArgumentParser
import json

from MCMC_utils import Replica, MarkovState, EnergyCallback, MarkovChain, LookUpTable, replica_exchange, calculate_state_energy, proposal_kernel, update_chain_dict, default_aa_space

ray.init()

def MH_step(step, stepsize, annealing_rate, exchange_interval, Beta, MC, EC, LUT):
  
    # obtain current state
    current_state = MC[-1]
    
    # decrease temp
    if step != 0:
        current_state.T *= annealing_rate

    # Replica exchange
    acceptance = False
    if (step % exchange_interval == 0) and (step != 0):
        accepted_exchanges = []
        exchange_state = deepcopy(current_state)
        if step % (2*exchange_interval) == 0:
            partners = [(i, i+1) for i in range(0, len(current_state)-1, 2)]
            for pair in partners:
                acceptance, exchange_state = replica_exchange(pair, exchange_state)
                accepted_exchanges.append(acceptance)
        else:
            partners = [(i, i+1) for i in range(1, len(current_state)-1, 2)]
            for pair in partners:
                acceptance, exchange_state = replica_exchange(pair, exchange_state)
                accepted_exchanges.append(acceptance)
        
        # update MC accepted chain and LUT
        if any(accepted_exchanges):
            acceptance = [True for repl in range(len(current_state))]
            MC.update_chain(acceptance, exchange_state)
            LUT.update_accepted_chain(MC)
            LUT.append(step, exchange_state, acceptance, MC)
        else:
            acceptance = [True for repl in range(len(current_state))]
            MC.update_chain(acceptance, current_state)
            LUT.update_accepted_chain(MC)
            LUT.append(step, current_state, acceptance, MC)
    
    # regular MH-step
    else:
        logging.info(f"MH-step at step: {step}")
        
        if step ==  0:
            for i in range(len(current_state)): 
                current_state.state[i].E = 5
        
        # propose new state and determine its energy
        proposed_state = proposal_kernel(current_state, stepsize)
        calculate_state_energy(proposed_state, EC, LUT) 
        
        # calculate state acceptance based on MH-criterion
        if all(p < c for p, c in zip(proposed_state.get_E(), current_state.get_E())):
            acceptance = [True for replica in range(len(proposed_state.state))]
        else:
            energies_new = np.array(proposed_state.get_E())
            energies_current = np.array(current_state.get_E())
            delta_E = energies_new - energies_current 
            p_accept = np.exp((-delta_E)*1/proposed_state.T*Beta)
            if step != 0:
                wandb.log({f'p_accept{rep}': np.clip(p_accept[rep], 0, 1) for rep in range(len(p_accept))}, step=step, commit=False)
            mc_prob = np.random.uniform(0,1,[len(proposed_state),])
            acceptance = [bool(mc < pa) for mc, pa in zip(mc_prob, p_accept)]

        # update MC accepted chain
        if any(acceptance):
            MC.update_chain(acceptance, proposed_state)
    
        # update LUT
        LUT.update_accepted_chain(MC)
        LUT.append(step, proposed_state, acceptance, MC)
    
    # wandb log step
    chain_dict = LUT.chain_metrics(step, MC)
    wandb.log(chain_dict, step=step)



def run_simulated_annealing(LUT):

    # initialize all global params
    seq = LUT.run_params["sequence"]
    mutate_res=LUT.run_params["mutate_res"]
    n_steps=LUT.run_params["n_steps"]
    annealing_rate=LUT.run_params["annealing_rate"]
    n_replicas=LUT.run_params["n_replicas"]
    exchange_interval=LUT.run_params["exchange_interval"]
    stepsize=LUT.run_params["stepsize"]
    T=LUT.run_params["temperature"]
    
    aa_space=default_aa_space
    Beta = [float(t) for t in np.linspace(1/n_replicas, 1, n_replicas)]
    
    EC = EnergyCallback(msa_tmp=LUT.run_params["msa_tmp_dir"], 
                targetDB=LUT.run_params["msa_targetDB"], 
                protname=LUT.run_params["protname"], 
                lig=LUT.run_params["ligand"], 
                cofac=LUT.run_params["cofactor"], 
                device=LUT.run_params["device"],
                cofactor_chain_idx=LUT.run_params["cofactor_chain_idx"],
                protein_chain_idx=LUT.run_params["protein_chain_idx"],
                af_input=LUT.run_params["af_input"],
                af_output=LUT.run_params["af_output"],
                models=LUT.run_params["models"],
                public_databases=LUT.run_params["public_databases"],
                alphafold3=LUT.run_params["alphafold3"],
                n_workers=LUT.run_params["n_workers"],
                state=None
                )
    
    # check if loading from a checkpoint or initializing a fresh chain
    if len(LUT.chain.keys()) == 0:
        logging.info(f'No previous Markov-States detected in LUT, initializing a new Markov-Chain...')
        MS = MarkovState(parent_seq=seq,
                         pos_space=mutate_res,
                         E=[100 for n in range(len(Beta))],
                         T=T,
                         Beta=Beta,
                         n_replicas=n_replicas,
                         aa_space=aa_space,
                         EnergyCallback=EC)
        MC = MarkovChain(n_replicas=n_replicas,
                        n_steps=n_steps)
        MC.initialize_chain(MS)
        for step in range(n_steps):
            MH_step(step, 
                    stepsize,
                    annealing_rate,
                    exchange_interval,
                    Beta,
                    MC,
                    EC,
                    LUT
                    )
        try:
            pass
        except Exception as exc:
            logging.info(f'Encountered and eception in main loop. {exc}')
            logging.info('Savely checkpointing current state.')
            LUT.atomic_save()
    else:
        logging.info(f'Detected Markov-States in LUT, loading from checkpoint.')
        MC = MarkovChain.from_dict(LUT, EC)
        try:
            checkpointed_step = len(MC)
            for step in range(checkpointed_step, n_steps):
                MH_step(step, 
                        stepsize,
                        annealing_rate,
                        exchange_interval,
                        Beta,
                        MC,
                        EC,
                        LUT
                        )
        except Exception as exc:
            logging.info(f'Encountered and eception in main loop. {exc}')
            logging.info('Savely checkpointing current state.')
            LUT.atomic_save()


@ray.remote(num_gpus=1)
def run_multi_GPU(device, LUT_dict):
    print(f"Running run_multi_GPU on device: {device}")
    #override device in run_params
    LUT_dict = deepcopy(LUT_dict)
    LUT_dict["run"]["device"] = device
    base, fname = os.path.splitext(LUT_dict["run"]["LUT_path"]) 
    directory = os.path.join(base, f'chain_{device}')
    os.makedirs(directory, exist_ok=True)
    LUT_dict["run"]["LUT_path"] = os.path.join(directory, fname)

    # initialize logging set up
    log_dir = LUT_dict["run"]["log_dir"]
    log_dir = eval(f"f'{log_dir}'") 
    log_dir = os.path.join(log_dir,
                            f'Chain_device_{device}') 
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, LUT_dict["run"]["log_name"])   
    
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[
                                logging.FileHandler(log_path),  
                                logging.StreamHandler()  
                                ]
                        )
    logger = logging.getLogger(__name__)

    # intialize wandb logging
    wandb.init(project="replica_exchange_MCMC",
                name=f"Chain_{device}",
                config={}
            )

    if LUT_dict:
        logging.info('Successfully loaded LUT_dict.json')
    
    # initialize LUT-object from LUT-dict
    LUT = LookUpTable.load(LUT_dict, device=device)

    # run simulated annealing
    run_simulated_annealing(LUT)


def run_single_GPU(LUT_dict):
        # initialize logging set up
        log_dir = LUT_dict["run"]["log_dir"]
        log_dir = eval(f"f'{log_dir}'") 
        log_path = os.path.join(log_dir, LUT_dict["run"]["log_name"])   
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s",
                            handlers=[
                                    logging.FileHandler(log_path),  
                                    logging.StreamHandler()  
                                    ]
                            )
        logger = logging.getLogger(__name__)

        # intialize wandb logging
        wandb.init(project="replica_exchange_MCMC",
                config={}
                )

        if LUT_dict:
            logging.info('Successfully loaded LUT_dict.json')
        
        # initialize LUT-object from LUT-dict
        LUT = LookUpTable.load(LUT_dict)

        # run simulated annealing
        run_simulated_annealing(LUT)


if __name__ == "__main__":

    # initialize parameters from json file
    parser = ArgumentParser()
    parser.add_argument("--LUT_json",
                    type=str,
                    required=False
                )
    args = parser.parse_args()
    LUT_path = args.LUT_json
    with open(LUT_path, 'r') as p:
        LUT_dict = json.load(p)

    # check whether multiple GPUs should be run
    if LUT_dict["run"]["device"] == 'all':
        # find all GPU device:
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        futures = [run_multi_GPU.remote(device, LUT_dict)
                   for device in range(n_gpus)
                ]
        ray.get(futures)
    else: 
        run_single_GPU(LUT_dict)
