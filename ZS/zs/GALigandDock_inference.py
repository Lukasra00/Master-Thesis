# Script to automatically run GALigandDock as a ZS
# Authors: Lukas Radtke {lradtke@caltech.edu}
# Date: 16/03/25

import os
import psutil
import pandas as pd
import multiprocessing
import argparse


from glob import glob
from copy import deepcopy
from tqdm import tqdm
from typing import Union

from concurrent.futures import ProcessPoolExecutor, as_completed

from  Master_Thesis.ZS.preprocess import ZSData

parser = argparse.ArgumentParser(
                    prog='Pyrosetta Pipeline',
                    description='This program prepares the inputs for PyRosetta GALigandDock. Type "-h" for help.'
)
parser.add_argument("--meta_string", 
                    type=str, 
                    help="Path to campaign metadata csv-file."
                    )
parser.add_argument("--meta_list", 
                    nargs='+', 
                    help="List of paths to campaign metadata csv-files spell out paths separated by a space."
                    )
parser.add_argument("--preprocessed_dir",
                    type=str,
                    help="Path to directory containing a folder for each campaign with preprocessed campaign pdb and param files."
                    )
parser.add_argument("--results_dir",
                    type=str,
                    help="Path to directory where a results folder for every campaign will be created."
                    )
args = parser.parse_args()
if args.meta_string is not None:
    pattern = args.meta_string
elif args.meta_list is not None:
    pattern = args.meta_list
else:
    print(f'ERROR: Meta csv file input not correctly spcified.')
kwargs = {"preprocessed_dir": args.preprocessed_dir,
          "results_dir": args.results_dir
        }


class PyrosettaData(ZSData):


    def __init__(
        self,
        input_csv: str,
        var_col_name: str = "var",
        mut_col_name: str = "mut",
        seq_col_name: str = "seq",
        fit_col_name: str = "fitness",
        preprocessed_dir: str ="data/preprocessed",
        results_dir: str = "/disk2/lukas/EnzymeOracle/data/GALigandDock_results"
    ):
        super().__init__(            
            input_csv=input_csv,
            var_col_name=var_col_name,
            mut_col_name=mut_col_name,
            seq_col_name=seq_col_name,
            fit_col_name=fit_col_name,
            )
        
        self.preprocessed_dir = preprocessed_dir
        self.results_dir = results_dir
        self.campaign_name = os.path.basename(self._input_csv)[:-4]
        self.results_folder_campaign = os.path.join(self.results_dir, self.campaign_name)
        self.param1_path = os.path.join(self.preprocessed_dir, self.campaign_name, self.campaign_name+'_ligand_corrected_unit1_am1bcc.params')
        self.param2_path = os.path.join(self.preprocessed_dir, self.campaign_name, self.campaign_name+'_ligand_corrected_unit2_am1bcc.params')

        self.parallel_inference()
        self.evaluate_docking()
        
    
    def parallel_inference(self):
        """
        Wrapper function for dynamically handling worker allocation and running GALigandDock_inference().
        """

        csv_path = self._input_csv
        campaign = os.path.split(csv_path)[-1][:-4]
        
    
        print(f'INFO: Starting GALigandDock inference for campaign {campaign}.')
        
        # determining available cores       
        core_percentages = psutil.cpu_percent(interval=1, percpu=True)
        num_workers_cpu = multiprocessing.cpu_count() - (30 + sum(percent > 5 for percent in core_percentages)) # leave a 10 core buffer
        
        # determining available memory and workers
        estimated_memory_per_process_bts = 200000 
        mem = psutil.virtual_memory()
        available_memory = mem.available
        num_workers_mem = available_memory // estimated_memory_per_process_bts
        num_workers = min(num_workers_cpu, num_workers_mem)
        if num_workers < 0: 
            num_workers = 10
            print(f'INFO: Less than 30 workers available. Initalizing with {num_workers} worker.')
        print(f'INFO: Parallel process pool initialized with {num_workers} workers.')
        
        # preparing inputs and running process pool
        preprocessed_pdb_filenames = [f for f in os.listdir(os.path.join(self.preprocessed_dir, campaign)) if f.endswith('final.pdb') and not f.endswith('_am1bcc.pdb') and not f.startswith(campaign)]
        preprocessed_pdb_filepaths = [os.path.join(self.preprocessed_dir, campaign, pdb_filename) for pdb_filename in preprocessed_pdb_filenames]
        params1_path = os.path.join(self.preprocessed_dir, campaign, campaign+'_ligand_corrected_unit1_am1bcc.params')
        params2_path = os.path.join(self.preprocessed_dir, campaign, campaign+'_ligand_corrected_unit2_am1bcc.params')
        results = []
        os.makedirs(self.results_folder_campaign, exist_ok=True)
        print(f'INFO: Created directory {self.results_folder_campaign} for results.')
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(self.GALigandDock_inference, pdb_filepath, params1_path, params2_path):  pdb_filepath for pdb_filepath in preprocessed_pdb_filepaths
            }

            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f'INFO: File {filename} generated an exception: {exc}')

                    
    def GALigandDock_inference(self, enzyme_final_pdb, param1_path, param2_path):
        """
        Iinitates pyrosetta and runs GALigandDock
        """

        import logging
        logging.basicConfig(level=logging.INFO)
        import os
        import pandas as pd
        import ipdb
        import pickle
        import pyrosetta
        import pyrosetta.distributed
        import pyrosetta.distributed.io as io
        import pyrosetta.distributed.viewer as viewer
        import pyrosetta.distributed.packed_pose as packed_pose
        import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
        import sys
        import logging
        logging.basicConfig(level=logging.INFO)

        flags = f"""
        -extra_res_fa {param1_path} {param2_path} \
        -gen_potential \
        -overwrite \
        -mute core basic protocols.relax \
        -crystal_refine -scale_rb 10.0 \
        -score::hb_don_strength hbdon_GENERIC_SC:1.45 \
        -score::hb_acc_strength hbacc_GENERIC_SP2SC:1.19 \
        -score::hb_acc_strength hbacc_GENERIC_SP3SC:1.19 \
        -score::hb_acc_strength hbacc_GENERIC_RINGSC:1.19 \
        -no_autogen_cart_improper \
        """
        pyrosetta.distributed.init(flags)
        pose_obj = io.pose_from_file(filename=enzyme_final_pdb)

        # Rosettascript
        xml = f"""
        <ROSETTASCRIPTS>
        <SCOREFXNS>
            <ScoreFunction name="dockscore" weights="beta_genpot">
            <Reweight scoretype="fa_rep" weight="0.2"/>
            <Reweight scoretype="coordinate_constraint" weight="0.1"/>
            </ScoreFunction>
            <ScoreFunction name="relaxscore" weights="beta_genpot_cart"/>
        </SCOREFXNS>
        <MOVERS>
            <GALigandDock name="dock"
                        runmode="dockflex"
                        scorefxn="dockscore"
                        scorefxn_relax="relaxscore"
                        grid_step="0.25"
                        padding="5.0"
                        hashsize="8.0"
                        subhash="3"
                        nativepdb="{enzyme_final_pdb}"
                        final_exact_minimize="bbsc"  
                        random_oversample="10"
                        rotprob="0.9"
                        rotEcut="100"
                        sidechains="aniso"
                        initial_pool="{enzyme_final_pdb}"
                        reference_pool="INPUT">
            <Stage repeats="15" npool="75" pmut="0.2" smoothing="0.375" rmsdthreshold="1.5" maxiter="50" pack_cycles="100" ramp_schedule="0.1,1.0"/>
            <Stage repeats="10" npool="25" pmut="0.2" smoothing="0.375" rmsdthreshold="1.2" maxiter="50" pack_cycles="100" ramp_schedule="0.1,1.0"/>
            </GALigandDock>
        </MOVERS>
        <PROTOCOLS>
            <Add mover="dock"/>
        </PROTOCOLS>
        <OUTPUT scorefxn="relaxscore"/>
        </ROSETTASCRIPTS>
        """
        ### final_exact_minimize="bbsc" -> now doing backbone and sidechain optimization

        xml_obj = rosetta_scripts.MultioutputRosettaScriptsTask(xml)
        xml_obj.setup()

        if not os.getenv("DEBUG"):
            results = list(xml_obj(pose_obj))


        # saving pickled pose and results df
        pose_results_path = os.path.join(self.results_folder_campaign, os.path.split(enzyme_final_pdb)[-1][:-4]+'.pkl')
        with open(pose_results_path, 'wb') as f:
            pickle.dump(results, f)
            print('Pickled result file successfully dumped!')
        df_results_path = os.path.join(self.results_folder_campaign, os.path.split(enzyme_final_pdb)[-1][:-4]+'.csv')
        if not os.getenv("DEBUG"):
            df = pd.DataFrame.from_records(packed_pose.to_dict(results))
            df.to_csv(df_results_path, index=False)
            print('Docked poses dataframe saved successfully!')
    

    def evaluate_docking(self):
        """
        Loads results tables and creates a summary df for a particular campaign.
        """

        eval_data = []
        meta_csv = pd.read_csv(self._input_csv)
        for idx, row in tqdm(meta_csv.iterrows(), desc=f'INFO: Evaluating docking results for {self.campaign_name}'):
            var = row[self._var_col_name]
            fitness = row['fitness']

            dock_csv_name = var.replace(':', '_').lower()+'_aligned_enzyme_final.csv'
            dock_csv_path = os.path.join(self.results_folder_campaign, dock_csv_name)
            try:
                dock_csv = pd.read_csv(dock_csv_path)
                score_list = dock_csv.columns.to_list()
                score_list.pop() # pop the pkl pose in the last col
                score_dict = {}
                for score_name in score_list:
                    score_dict[score_name] = min(dock_csv[score_name].to_list())
                row = [var, fitness] + [score for score in score_dict.values()]
                eval_data.append(row)
            except FileNotFoundError:
                print(f'ERROR: File not found {dock_csv_path}. Skipping variant...')
        column_list = ['var', 'fitness'] + score_list
        eval_df = pd.DataFrame(eval_data, columns=column_list)

        eval_df_path = os.path.join(self.results_dir, self.campaign_name+'.csv')
        eval_df.to_csv(eval_df_path)
        if os.path.exists(eval_df_path):
            print(f'INFO: Succesfully saved evaluation df to {eval_df_path}.')
        else:
            print(f'ERROR: Failed to save evaluation df to {eval_df_path}.')
        

def run_pyrosetta_pipeline(pattern: Union[str, list] = None, kwargs: dict = {}):
    if isinstance(pattern, str):
        lib_list = glob(pattern)
    else:
        lib_list = deepcopy(pattern)

    for lib in lib_list:
        print(f"Running Pyrosetta Pipeline for {lib}...")
        PyrosettaData(input_csv=lib, **kwargs)


run_pyrosetta_pipeline(pattern=pattern, kwargs=kwargs)

# python -m REVIVAL.zs.pyrosetta_inference --meta_list /disk2/fli/REVIVAL2/data/meta/ParLQ-d.csv /disk2/fli/REVIVAL2/data/meta/ParLQ-e.csv /disk2/fli/REVIVAL2/data/meta/ParLQ-f.csv /disk2/fli/REVIVAL2/data/meta/ParLQ-g.csv /disk2/fli/REVIVAL2/data/meta/ParLQ-h.csv /disk2/fli/REVIVAL2/data/meta/ParLQ-i.csv --preprocessed_dir /disk2/lukas/EnzymeOracle/data/preprocessed --results_dir /disk2/lukas/EnzymeOracle/data/GALigandDock_results
