# File to automatically run Placer as a ZS
# Authors: Lukas Radtke {lradtke@caltech.edu}
# Contributions:
# - LR: code prototyping and initial implementation
# Date: 21/06/25

import os 
import numpy as np
import pandas as pd
import subprocess
import multiprocessing
import psutil
from tqdm import tqdm
from copy import deepcopy
from glob import glob
from scipy import stats 
from concurrent.futures import ProcessPoolExecutor, as_completed

from Master_Thesis.ZS.preprocess import ZSData


class PlacerZS(ZSData):
    def __init__(self,
            input_csv: str,
            struc_dir: str,
            out_dir: str ,
            placer_dir: str,
            n_inferences: int = 10,
            ligand_1: str = 'LG1',
            ligand_2: str = 'LG2',
            fit_col_name: str = 'fitness',
            var_col_name: str = 'var'
            ):

        super().__init__(
            input_csv=input_csv,   
            fit_col_name = fit_col_name,
            var_col_name = var_col_name
        )
        self.input_csv=input_csv
        self.out_dir = out_dir
        self.placer_dir = placer_dir
        self.n_inferences = n_inferences
        self.ligand_1 = ligand_1
        self.ligand_2 = ligand_2

        self.campaign = os.path.basename(input_csv)[:-4]
        self.struc_dir = os.path.join(struc_dir, self.campaign)

        self.placer_parallel_inference()
        self.evaluate_placer()
                
    def placer_inference(self, variant):
        variant_path = os.path.join(self.struc_dir, variant)      
        out_dir = os.path.join(self.out_dir, self.campaign)
        os.makedirs(out_dir, exist_ok=True)
        print(f'USING variant {variant_path}')
        assert os.path.exists(variant_path), f'{variant_path} does not exist.'
        try:
            cli = ['python',
                os.path.join(self.placer_dir, 'run_PLACER.py'),
                '--ifile',
                variant_path,
                '--odir',
                out_dir,
                '--rerank',
                'prmsd',
                '--nsamples',
                str(self.n_inferences),
                '--predict_ligand',
                self.ligand_1,
                self.ligand_2,
                '--predict_multi'                   
            ]
            _ = subprocess.run(cli)
        except:
            print(f'############ Failed on variant {variant_path}')



    def placer_parallel_inference(self):
        meta = pd.read_csv(self.input_csv)
        var_names = [var_col.replace(':', '_').lower() for var_col in meta[self._var_col_name].to_list()]

        variant_list = [var+'_aligned_enzyme_final.pdb' for var in var_names ]

        # determining available cores       
        core_percentages = psutil.cpu_percent(interval=1, percpu=True)
        num_workers_cpu = multiprocessing.cpu_count() - (30 + sum(percent > 5 for percent in core_percentages)) # leave a 10 core buffer
        
        # determining available memory and workers
        estimated_memory_per_process_bts = 350000000 
        mem = psutil.virtual_memory()
        available_memory = mem.available
        num_workers_mem = available_memory // estimated_memory_per_process_bts
        num_workers = min(num_workers_cpu, num_workers_mem)
        if num_workers < 0: 
            num_workers = 10
            print(f'INFO: Less than 30 workers available. Initalizing with {num_workers} worker.')
        elif num_workers > len(variant_list):
            num_workers = len(variant_list)
            print(f'INFO: Parallel process pool initialized with {num_workers} workers.')
        num_workers = 10

        results = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(self.placer_inference, variant): variant for variant in tqdm(variant_list, desc='Generating Ensembles.')
            }
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f'INFO: File {filename} generated an exception: {exc}')



    def evaluate_placer(self):
        meta = pd.read_csv(self.input_csv)
        result_list = []
        missing_path_counter = 0
        missing_paths = []
        for i, variant in meta.iterrows():
            var = variant[self._var_col_name]
            fitness = variant[self._fit_col_name]

            # load corresponding placer output
            try:
                variant_path = os.path.join(self.out_dir, self.campaign, var.replace(':', '_').lower()+'_aligned_enzyme_final.csv')
                variant_csv = pd.read_csv(variant_path)
                fape = variant_csv['fape'].to_list()
                fape_avg = np.mean(fape)
                fape_SD = np.std(fape)

                lddt = variant_csv['lddt'].to_list()
                lddt_avg = np.mean(lddt)
                lddt_SD = np.std(lddt)

                rmsd = variant_csv['rmsd'].to_list()
                rmsd_avg = np.mean(rmsd)
                rmsd_SD = np.std(rmsd)

                kabsch = variant_csv['kabsch'].to_list()
                kabsch_avg = np.mean(kabsch)
                kabsch_SD = np.std(kabsch)

                prmsd = variant_csv['prmsd'].to_list()
                prmsd_avg = np.mean(prmsd)
                prmsd_SD = np.std(prmsd)

                plddt = variant_csv['plddt'].to_list()
                plddt_avg = np.mean(plddt)
                plddt_SD = np.std(plddt)

                plddt_pde = variant_csv['plddt_pde'].to_list()
                plddt_pde_avg = np.mean(plddt_pde)
                plddt_pde_SD = np.std(plddt_pde)

                variant_list = [var,
                                fitness,
                                fape_avg,
                                fape_SD,
                                lddt_avg,
                                lddt_SD,
                                rmsd_avg,
                                rmsd_SD,
                                kabsch_avg,
                                kabsch_SD,
                                prmsd_avg,
                                prmsd_SD,
                                plddt_avg,
                                plddt_SD,
                                plddt_pde_avg,
                                plddt_pde_SD
                ]

                result_list.append(variant_list)
            except FileNotFoundError:
                missing_path_counter += 1
                print(f'ERROR: {missing_path_counter} could not find file {variant_path}')
                missing_paths.append(variant_path)
        

        header =  ['var',
                'fitness',
                'fape_avg',
                'fape_SD',
                'lddt_avg',
                'lddt_SD',
                'rmsd_avg',
                'rmsd_SD',
                'kabsch_avg',
                'kabsch_SD',
                'prmsd_avg',
                'prmsd_SD',
                'plddt_avg',
                'plddt_SD',
                'plddt_pde_avg',
                'plddt_pde_SD'
            ]
        result_df = pd.DataFrame(result_list, columns=header)
        results_df_path = os.path.join(self.out_dir, self.campaign+'_results.csv')
        result_df.to_csv(results_df_path)

        assert result_df.shape[0] == meta.shape[0], f"ERROR: meta and results shape does not match {result_df.shape[0]} and {meta.shape[0]}. You might be missing results."
        

        # print out the spearmanrho
        fitness_list = result_df[self._fit_col_name].to_list()
        for head in header:
            col_list = result_df[head].to_list()

            rho = stats.spearmanr(fitness_list, col_list)[0]
            print(f'Spearmancorr of {head}: {rho}')
        
            


def run_placer(pattern: str | list = None, kwargs: dict = {}) -> None:
    if isinstance(pattern, str):
        lib_list = sorted(glob(pattern))
    else:
        lib_list = deepcopy(pattern)

    for lib in tqdm(lib_list):
        print(f"Running PLACER for {lib}...")
        PlacerZS(input_csv=lib, **kwargs)


# Example run cmds
csv_list = [
    '.../Rma-CSi.csv',
    '.../Rma-CB.csv'
]
kwargs = {'struc_dir':'.../dummy_placer/'}
run_placer(pattern=csv_list, kwargs=kwargs)

