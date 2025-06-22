from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from scipy import stats
import json
import subprocess
import time
import os
import HTSeq
import random
import shutil
import ipdb
import tempfile

from concurrent.futures import ProcessPoolExecutor, as_completed, default_aa_space


class Replica():
    def __init__(self, seq, E, beta):
        self.seq = seq
        self.E = E
        self.beta = beta

    def to_dict(self):
        return {'seq': self.seq, 
                'E': self.E, 
                'beta': self.beta}
    
    @classmethod
    def from_dict(cls, d):
        return cls(seq=d['seq'], 
                   E=d['E'], 
                   beta=d['beta'])
    
    def __str__(self):
        return f'Replica object:\nEnergie: {self.E}\nBeta Temperature factor: {self.beta}'
    

class MarkovState():
    def __init__(self, parent_seq, pos_space, E, T, Beta, n_replicas, aa_space, EnergyCallback):
        self.parent_seq = parent_seq
        self.pos_space = pos_space
        self.aa_space = aa_space
        self.T = T
        self.Beta = Beta
        self.n_replicas = n_replicas
        self.state = [Replica(seq=self.parent_seq,
                              E=Er,
                              beta=Br) for Er, Br in zip(E, self.Beta)]
        self.EnergyCallback = EnergyCallback
        assert len(self.Beta) == self.n_replicas == len(E), logging.info('Error: Beta, E and n_replicas are expected to be of same dimensionality.')
    
    def to_dict(self):
        return {
            'parent_seq':   self.parent_seq,
            "pos_space":  self.pos_space,      
            "aa_space":   self.aa_space,       
            'T':     self.T,
            'Beta':  self.Beta,
            'state': [rep.to_dict() for rep in self.state],
        }
    
    @classmethod
    def from_dict(cls, state_dict, EnergyCallback):
        obj = cls(parent_seq=state_dict['parent_seq'],
                  pos_space=state_dict["pos_space"],         
                  T=state_dict['T'],
                  E=[state_dict["state"][i]['E'] for i in range(len(state_dict["state"]))],
                  Beta=state_dict['Beta'],
                  n_replicas=len(state_dict['state']),
                  aa_space=state_dict["aa_space"],
                  EnergyCallback=EnergyCallback)
        obj.state = [Replica.from_dict(r) for r in state_dict['state']]
        return obj
    
    def __len__(self):
        return self.n_replicas
    
    def __setitem__(self, replica_idx, replica):
        self.state[replica_idx] = replica

    def __getitem__(self, idx):
        return self.state[idx]
    
    def __str__(self):
        string = f'<Markov State Object with {self.n_replicas} Replicas>\nSequence: {self.seq}\nSampling space:{self.pos_space}\nBeta vector: {self.Beta}\nEnergies: {self.get_E()}'
        return string  
    
    def add_correlated_column(self, df, old_col, new_col_name, spearman_rho, target_sd):
        X = df[old_col]
        n = len(X)
        ranks = stats.rankdata(X)       
        U = (ranks - 0.5) / n           
        X_norm = stats.norm.ppf(U)
        r_pearson = 2 * np.sin(np.pi * spearman_rho / 6)
        noise = np.random.randn(n)
        Y_norm = r_pearson * X_norm + np.sqrt(1 - r_pearson**2) * noise
        Y_scaled = Y_norm * target_sd / np.std(Y_norm)
        df[new_col_name] = Y_scaled
        return df
    
    def update_state_energies_dummy(self):
        # dummy function, should later be replaced with ZS score, i.e. alphafold3 calling
        df_path = '/disk2/lukas/EnzymeOracle/data/meta/TrpB4.csv'
        wt_seq = "MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARRLSEKYGARVYLKREDLLHTGAHKINNAIGQVLLAKLMGKTRIIAETGAGQHGVATATAAALFGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVKSGSRTLKDAIDEALRDWITNLQTTYYVFGSVVGPHPYPIIVRNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAAGIFYPFIDSGVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGSKTFVLQDDWGQVQVSHSVSAGLDYSGVGPEHAYWRETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVVNLSGRGDKDLESVLNHPYVRERIRL",
        mutate_res = [183]

        df = pd.read_csv(df_path)
        spearman_rho = 0.15 # empirical std over the pAE ZS of the MCMC mmseqs2 LUT
        fitness_SD = np.std(df['fitness']) + 2
        df_wZS = self.add_correlated_column(df, 'fitness', 'ZS_fitness', spearman_rho, fitness_SD)
        
        for replica_idx in range(self.n_replicas):
            seq = self.state[replica_idx].seq
            mutate_res = [mut+1 for mut in mutate_res]
            muts = [seq[mutate_res[i]] for i in range(len(mutate_res))]
            muts = ''.join(muts)
            try:
                ZS_fitness = float(df_wZS.loc[df_wZS['AAs'] == muts, 'ZS_fitness'])
            except: 
                logging.info(f'Could not find ZS for variant {muts}. Setting fitness to 0.')
                ZS_fitness = 0 
            self.state[replica_idx].E = ZS_fitness

    def get_E(self):
        return [self.state[replica_idx].E for replica_idx in range(self.n_replicas)]


class MarkovChain():
    def __init__(self, n_replicas, n_steps):
        self.chain = []
        self.n_replicas = n_replicas
        self.n_steps = n_steps

    def to_dict(self):
        return {
            'n_replicas':  self.n_replicas,
            'n_steps':     self.n_steps,
            'chain':       [st.to_dict() for st in self.chain],
        }

    @classmethod
    def from_dict(cls, LUT, EnergyCallback):
        mc = cls(n_replicas=LUT.run_params['n_replicas'],
                 n_steps   =LUT.run_params['n_steps'])

        state_list = (LUT.accepted_chain["chain"]
                      if isinstance(LUT.accepted_chain, dict)
                      else LUT.accepted_chain)

        mc.chain = [MarkovState.from_dict(sd, EnergyCallback)
                    for sd in state_list]
        return mc

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            i, j = idx
            state = self.chain[i]
            replica = state[j]
            return replica
        else:
            return self.chain[idx]
    
    def __str__(self):
        return f'<Markov Chain Object>\nLength: {len(self.chain)}\nReplicas: {self.n_replicas}\nMax steps: {self.n_steps}'
    
    def __len__(self):
        return len(self.chain)

    def initialize_chain(self, state):
        self.chain.append(state)

    def update_chain(self, state_acceptance, proposed_state):
        if isinstance(state_acceptance, bool):
            pass
        else:
            new_state = deepcopy(self.chain[-1])
            for replica_idx in range(self.n_replicas):
                if state_acceptance[replica_idx]:
                    new_state[replica_idx] = proposed_state[replica_idx]  # Correct: assign only the matching replica
                else:
                    new_state[replica_idx] = self.chain[-1][replica_idx]
            self.chain.append(new_state)


class EnergyCallback():
    def __init__(self, state, msa_tmp, targetDB, protname, lig, cofac, cofactor_chain_idx, protein_chain_idx, device, af_input, af_output, models, public_databases, alphafold3, n_workers):
        self.protname = protname.lower()
        self.lig = lig
        self.cofac = cofac
        self.cofactor_chain_idx = cofactor_chain_idx
        self.protein_chain_idx = protein_chain_idx
        self.device = device
        self.af_input = af_input
        self.af_output = af_output
        self.models =  models
        self.public_databases = public_databases
        self.alphafold3 = alphafold3
        self.af3_seed = 42
        self.msa_tmp = msa_tmp
        self.targetDB = targetDB  
        self.state = state 
        self.n_workers = n_workers
    
    def query_structure_in_LUT(self, sequence):
        pass
        # test whether the seq has already been calculated energy for

    def generate_MSA_parallel(self, precomputed):
        # take proposals for the replicas and generate MSAs

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            not_precomputed_repl_idx = [i for i, val in enumerate(precomputed) if not val]
            futures = [executor.submit(self.generate_MSA,
                                       replica=self.state[repl],
                                       ) for repl in not_precomputed_repl_idx
                    ]                                                        
        json_paths = {}
        for future in as_completed(futures):
            try:
                beta, path = future.result()
                json_paths[beta] = path
            except Exception as e:
                logging.error(f"Error in a worker during parallel MSA: {e}")
        
        return json_paths

    def generate_MSA(self, replica):
        beta = replica.beta
        seq = replica.seq
        seed_value = int(time.time() * 1_000_000)
        random.seed(seed_value)
        seed = random.randint(0, 2**31)
        jobname = str(seed)
        af3_input_json_path = os.path.join(self.af_input, self.protname+'_'+jobname+'.json')

        MSAstart_time = time.perf_counter()
        
        # generate Fasta from query sequence
        seq_bytes = seq.encode('utf-8')
        seq_obj = HTSeq.Sequence(seq_bytes, name=f"query_job_{jobname}")
        msa_job_folder = os.path.join(self.msa_tmp, jobname)
        os.makedirs(msa_job_folder, exist_ok=True)
        msa_fasta_path = os.path.join(msa_job_folder, f'query_{jobname}.fasta')
        fasta_file = open(msa_fasta_path, 'w')
        seq_obj.write_to_fasta_file(fasta_file)
        fasta_file.close()

        # generate queryDB from Fasta
        queryDB_path = os.path.join(msa_job_folder, 'query_db')
        os.makedirs(msa_job_folder, exist_ok=True)
        generate_queryDB = [
            "mmseqs",
            "createdb", 
            msa_fasta_path,
            queryDB_path
        ]
        try:
            logging.info(f'Starting MSA...')
            logging.info(f'Generating queryDB for job: {jobname}, beta: {beta}')
            out = subprocess.run(generate_queryDB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            logging.error(f'An exception occured in: {exc}')
        
            
        # search the targetDB with the queryDB
        tmp_search_path = os.path.join(msa_job_folder, 'tmp')
        os.makedirs(tmp_search_path, exist_ok=True)  
        results_path = os.path.join(msa_job_folder, 'results','resultsDB')
        os.makedirs(os.path.dirname(results_path))
        print(f"queryDB: {queryDB_path}\ntarget DB: {self.targetDB}\nresults_path: {results_path}\n tmp:{tmp_search_path}")
        MSAsearch = [
            'mmseqs',
            'search',
            queryDB_path,
            self.targetDB,
            results_path,
            tmp_search_path,
            '--gpu',
            '1',
            '--max-seqs',
            '20000',
            '-s',
            '1'
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.device)
        try:
            logging.info(f'Searching the targetDB with queryDB for job: {jobname}, beta: {beta}') 
            out = subprocess.run(MSAsearch, env=env)
        except Exception as exc:
            logging.error(f'An exception occured in: {exc}')

        # clear tmp search path
        for item in os.listdir(tmp_search_path):
            item_path = os.path.join(tmp_search_path, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
            except Exception as exc:
                print(f'Cleaning of tmp dir with rmtree lead to an exception: {exc}')
        
        # foldseek reformat as a3mMSA
        msa_out_path = os.path.join(os.path.dirname(results_path), 'resultMSA')
        a3mMSA = [
            'foldseek',
            'result2msa',
            queryDB_path,
            self.targetDB,
            results_path,
            msa_out_path,
            '--msa-format-mode',
            '6',
        ]
        try:
            logging.info(f'Converting resultDB to a3m format for job: {jobname}, beta: {beta}')
            out = subprocess.run(a3mMSA, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            logging.error(f'An exception occured in: {exc}')
        
        try:
            logging.info(f'MSA is loaded and added to input json for job: {jobname}, beta: {beta}')
            with open(msa_out_path, 'r', encoding='utf-8') as f:
                unpairedMSA = f.read()
        except Exception as exc:
            logging.error(f'An exception occured in: {exc}')


        # remove escape character that screws up af3
        unpairedMSA = unpairedMSA.rstrip("\u0000")

        # generate af3 input file with MSA

        input_json = {
            "name": self.protname,
            "sequences": [
            {
                "protein": {
                "id": "A",
                "sequence": f'{seq}',
                "unpairedMsa": f"{unpairedMSA}",
                "pairedMsa": "", 
                "templates": []
                }
            },
            {
                "ligand": {
                "id": "L",
                "smiles": self.lig
                }
            },
            {         
                "ligand": {
            "id": "C",
            "smiles": self.cofac  
            }
            }
            ],
            "modelSeeds": [42],
            "dialect": "alphafold3",
            "version": 1  
        }
    

        try:
            with open(af3_input_json_path, 'w') as f:
                json.dump(input_json, f, indent=4) 
            logging.info(f'Successfully dumped AF3 input for job:  {jobname}, beta: {beta}')
        except:
            logging.error(f'Could not dump AF3 input for job: {jobname}, beta: {beta} \nat {af3_input_json_path}')

        MSAfinish_time = time.perf_counter()
        MSAduration = MSAfinish_time - MSAstart_time 
        logging.info(f'Finished MSA for job: {jobname}, beta: {beta} in {MSAduration:.2f} seconds.')
        return beta, af3_input_json_path

        
    def fold(self, input_json_path):        
        # mounted filepaths
        jobname = os.path.basename(input_json_path).split('_')[-1][:-5]
        af_output = os.path.join(self.af_output, jobname)

        fold_start_time = time.perf_counter()
        logging.info(f'Input file passed to fold method for job: {jobname}')
        fold_cmd = [
            'docker',
            'run',
            '-it',
            '--volume',
            f'{self.af_input}:/root/af_input',
            '--volume',
            f'{af_output}:/root/af_output',
            '--volume',
            f'{self.models}:/root/models',
            '--volume',
            f'{self.public_databases}:/root/public_databases',
            '--volume',
            f'{self.alphafold3}:/root/af_input/alphafold3',
            '--gpus',
            f'device={self.device}',
            'alphafold3', # name of the docker image, docker will look for this one locally
            'python',
            '/root/af_input/alphafold3/run_alphafold.py',
            f'--json_path=/root/af_input/{self.protname}_{jobname}.json',
            '--jackhmmer_n_cpu=8', 
            '--nhmmer_n_cpu=8',
            '--model_dir=/root/models',
            '--output_dir=/root/af_output',
            '--run_data_pipeline=False', 
            '--run_inference=True',
            '--buckets',
            '256'
        ]
        try: 
            logging.info(f'Running AF3 folding for job: {jobname}')
            out = subprocess.run(fold_cmd,  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception as exc:
            logging.error(f'An exception occured during AF3 folding of {self.protname}_{jobname}.json\nexc: {exc}')
        fold_finish_time = time.perf_counter()
        fold_duration = fold_finish_time - fold_start_time 
        logging.info(f'Finished folding for job {jobname} in {fold_duration:.2f} seconds.')

        # output handling, adjust filepaths as needed 
        struc_path = os.path.join(af_output, self.protname, 'seed-'+str(self.af3_seed)+'_sample-0', 'model.cif')
        confidences_path = os.path.join(af_output, self.protname, f'{self.protname}_summary_confidences.json')
        return struc_path, confidences_path

    def extract_metric(self, summary_confidences_path):
        with open(summary_confidences_path, 'r') as f:
            confidences = json.load(f)
        #score = confidences["chain_pair_pae_min"][self.cofactor_chain_idx][self.protein_chain_idx]
        score = confidences["chain_pair_pae_min"][1][self.protein_chain_idx] + confidences["chain_pair_pae_min"][2][self.protein_chain_idx]
        logging.info(f'Energy score: {score}')
        return score 
     
def proposal_kernel(current_state, stepsize):
    # generate random seed 
    try:
        t_ms = int(time.time() * 1e3)
        seed = (t_ms + 1) & 0xFFFFFFFF
        np.random.seed(seed)
    except Exception as exc:
        logging.info(f'Encountered exception during seed-generation.')
    
    # make a random step of stepsize in sequence space
    assert len(stepsize) == current_state.n_replicas, logging.info('Error: Stepsize vector dimensionality and n_replicas should match.')
    proposed_state = deepcopy(current_state)
    for replica_idx in range(len(current_state)):
        n_steps = stepsize[replica_idx]
        for _ in range(n_steps):
            n_mutatable_res = len(current_state.pos_space)
            rand_pos_idx = np.random.randint(0, n_mutatable_res)
            mut_pos = current_state.pos_space[rand_pos_idx]
            old_aa = current_state[replica_idx].seq[mut_pos-1]
            valid_aa_choices = [aa for aa in current_state.aa_space if aa != old_aa]
            rand_aa_idx = np.random.randint(0, len(valid_aa_choices))
            new_aa = valid_aa_choices[rand_aa_idx]
            proposed_state[replica_idx].seq = proposed_state[replica_idx].seq[:mut_pos-1] + new_aa + proposed_state[replica_idx].seq[mut_pos:]
    return proposed_state

def replica_exchange(pair, current_state):

    Ei = np.array(current_state.get_E()[pair[0]])
    Ej = np.array(current_state.get_E()[pair[1]])

    beta_i = current_state.state[pair[0]].beta
    beta_j = current_state.state[pair[1]].beta

    log_ratio = (beta_i - beta_j) * (Ej - Ei)
    p_accept_exchange  = min(1.0, np.exp(log_ratio))
    mc_prob = np.random.uniform(0,1)
    if mc_prob < p_accept_exchange:
        new_state = deepcopy(current_state)
        new_state.state[pair[0]].beta = deepcopy(current_state.state[pair[1]].beta)
        new_state.state[pair[1]].beta = deepcopy(current_state.state[pair[0]].beta)
        accepted = True
        logging.info(f'Accepted replica exchange between state idx {pair[0]} and {pair[1]} Exchange prob: {np.exp(log_ratio)}.')
        return accepted, new_state
    else:
        accepted = False
        logging.info(f'Rejected replica exchange between state idx {pair[0]} and {pair[1]}.')
        return accepted, current_state

def update_chain_dict(MarkovChain, accepted):
    chain_dict = {}
    for replica_idx in range(len(MarkovChain.chain[-1])):
        beta = MarkovChain.chain[-1][replica_idx].beta        
        chain_dict[f'Replica-Chain_Beta_{beta}'] = MarkovChain.chain[-1].get_E()[replica_idx]
        chain_dict["Temperature"] = MarkovChain[-1].T
        chain_dict["Average State Energy"] = np.mean(MarkovChain[-1].get_E())
        chain_dict["Best State Energy"] = np.min(MarkovChain[-1].get_E())
        #chain_dict["Exchange Acceptance Rate"] = n_ex_acc / (n_ex_acc + n_ex_decl + 1e-06)
        if accepted:
            accepted_numeric = 1
        else:
            accepted_numeric = 0
        chain_dict["Accepted Exchanges"] = accepted_numeric
    return chain_dict


def calculate_state_energy(proposed_state, EC, LUT):
    # checking if the state`s energy has already been computed in the past
    precomputed = [False for repl in range(len(proposed_state))]
    logging.info('Checking for Sequence in LUT')
    for replica_idx_1 in range(len(proposed_state)):
        query_seq = proposed_state[replica_idx_1].seq
        for step in LUT.chain.keys():
            for replica_idx_2 in range(len(proposed_state)):
                LUT_seq = LUT.chain[str(step)]["state"][replica_idx_2]["seq"]
                if query_seq == LUT_seq:
                    logging.info('Found pre-calculated energy for query sequence in LUT.')
                    proposed_state[replica_idx_1].E = LUT_seq = LUT.chain[str(step)]["state"][replica_idx_2]["E"]
                    precomputed[replica_idx_1] = True

    # if sequence is novel, use EnergyFunction Callback 
    EC.state = proposed_state
    json_paths = EC.generate_MSA_parallel(precomputed)
    for beta in json_paths.keys():
        _, confidences_path = EC.fold(json_paths[beta])
        score = EC.extract_metric(confidences_path)
        for replica in proposed_state.state:
                if replica.beta == beta:
                    replica.E = score
                    logging.info(f'Successfully calculated energy for replica beta={beta}')

class LookUpTable():
    def __init__(self, run_params, chain, accepted_chain, device=None):
        self.run_params=run_params
        self.chain=chain
        self.accepted_chain=accepted_chain
        self.device = device
    
    def update_accepted_chain(self, MC):
        # overrides the accepted Markov chain 
        MC_dict = MC.to_dict()
        self.accepted_chain = MC_dict
        try:       
            self.atomic_save()
        except Exception as exc:
            logging.info(f'Encountered and exception during atomic writing operation {exc}')                          
    
    def append(self, step: int, markov_state: MarkovState, accepted: list[bool], MC):
        payload=markov_state.to_dict()
        payload["accepted"]=accepted
        self.chain[str(step)]=payload
        try:       
            self.atomic_save()
        except Exception as exc:
            logging.info(f'Encountered and exception during atomic writing operation {exc}')                          
    
    def atomic_save(self):
        root, fname = os.path.split(self.run_params["LUT_path"])
        os.makedirs(root, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w',
                                        dir=root,
                                        prefix=fname,
                                        suffix='.tmp',
                                        delete=False
            ) as tf:
            json.dump({"run": self.run_params,
                    "chain": self.chain,
                    "accepted_chain": self.accepted_chain
                    }, 
                    tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
        iteration = 0
        if len(self.chain.keys()) % self.run_params["backup_every_n"] == 0:
            iteration = len(self.chain.keys())
        root, ext = os.path.splitext(self.run_params["LUT_path"])
        out_path = root + '_backup_' + str(iteration) + ext
        os.replace(tf.name, out_path)

    @classmethod
    def load(cls, LUT_dict, device=None):
        return cls(run_params=LUT_dict['run'],
                chain=LUT_dict['chain'],
                accepted_chain=LUT_dict['accepted_chain'],
                device=device)
    
    def chain_metrics(self, step, MC):
        chain_dict = {}
        
        # metrics dependent on last state in Markov chain:
        for beta in MC[-1].Beta:
            for i in range(len(MC[-1])):
                if MC[-1].state[i].beta == beta:
                    chain_dict[f"Replica Energy Beta={beta}"] = MC[-1].state[i].E
        chain_dict['Average State Energy'] = np.mean([MC[-1].state[i].E for i in range(self.run_params['n_replicas'])])
        chain_dict['Best State Energy'] = np.min([MC[-1].state[i].E for i in range(self.run_params['n_replicas'])])
        chain_dict['Sampling Temperatur'] = MC[-1].T

        # quantify last 5 state acceptance rate for each chain
        return chain_dict
    
