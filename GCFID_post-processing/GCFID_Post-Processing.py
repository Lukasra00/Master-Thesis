"""
This script postprocesses raw data of a gas chromatography flame ionization detection device (GCFID).
Date: 25.05.2025
Author: Lukas Radtke <lradtke@calteche.edu>

INPUTS in the json [key: description]:
- "balrog_basedir": Basedirectory of the analysis-run
- "run_name": Name of the analysis-run
- "parent_wells": List of wells, were the parent mutant was present 
- "out_dir": Basedirectory were outputs shall be stored.
- "starting_material": Column retention time of starting material [min] 
- "internal_standard": Column retention time of internal standard [min] 
- "product": Column retention time of product material [min] 
- "visualization_partial": Outpath of levseq streamlit visualization partial ->see: 
- "ZS_table": The correlation df to which to append the ground truth fitness values,

OUTPUTS:
- Plate array (.csv) that holds sequence and fitness data
- Levseq Streamlit app visualization partial 

"""
import os
import ipdb
import pandas as pd
import numpy as np
import argparse
import json
import typing

class EvalBalrog():
    def __init__(self, 
                 balrog_basedir: str,
                 run_name: str,
                 parent_wells: list[str],
                 out_dir: str,
                 starting_material: float,
                 internal_standard: float,
                 product: float,
                 visualization_partial: str,
                 ZS_table: str 
                 ):
        
        self.balrog_basedir = balrog_basedir
        self.run_name = run_name              # must match the plate 'name' column in levseq visualization partial
        self.parent_wells = parent_wells
        self.out_dir = out_dir
        self.starting_material = starting_material
        self.internal_standard = internal_standard
        self.product = product
        self.visualization_partial = visualization_partial
        self.ZS_table = ZS_table

        # create a plate map
        rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
        self.plate_map = {}
        well_number = 1
        for row in rows:
            for col in range(1, 13):
                self.plate_map[f"{row}{col:02d}"] = {}
                well_number += 1

        # data headers:
        self.data_headers = {
        "Peak Number": 0, 
        "Retention Time": 1,
        "Peak Type": 2,
        "Peak Width": 3,
        "  Area  ": 4,
        "Height": 5,
        "  Area   %": 6
        }
        
        # adds all raw GCFID filepaths to the platemap
        self.add_paths_to_platemap()
   
    def add_paths_to_platemap(self):
        """
        For each well, adds the raw data paths of the GCFID to the plate map dict.  
        """
        for folder in os.listdir(self.balrog_basedir):
            if folder[-5:-2] in self.plate_map.keys():
                well_path = os.path.join(self.balrog_basedir, folder)
                meta_path = os.path.join(self.balrog_basedir, folder, 'Report00.CSV')
                data_path = os.path.join(self.balrog_basedir, folder, 'REPORT01.CSV')
                self.plate_map[folder[-5:-2]]['well_path'] = well_path
                self.plate_map[folder[-5:-2]]['meta_path'] = meta_path
                self.plate_map[folder[-5:-2]]['data_path'] = data_path

    def seq_map(self):
        """
        Takes mutations from the visualization partial and writes to sequence map.
        Collects all parent wells.
        """
        vis_par = pd.read_csv(self.visualization_partial)
        for well in self.plate_map.keys():
            well_short = self.well_format_converter(well)
            var = vis_par.loc[(vis_par["Well"] == well_short) & (vis_par["name"] == self.run_name), "amino_acid_substitutions"].item()
            self.plate_map[well]["variant"] = var 
        self.vars = [str(self.plate_map[well]['variant']) for well in self.plate_map.keys()]
        # gather all parent_wells of the plate
        self.parent_wells = []
        for well in self.plate_map.keys():
            var = self.plate_map[well]["variant"]
            if var == '#PARENT#':
                self.parent_wells.append(well)

    def find_peak(self, 
                  retention_time: float, 
                  data_headers: list[str], 
                  df: pd.DataFrame):
        """
        Finds the GCFID peak for a expected retention time (within a +/- 0.05min range).
        """
        peak_idx = (df[data_headers["Retention Time"]] - retention_time).abs().idxmin()
        exact_peak_retention = df.loc[peak_idx ,data_headers["Retention Time"]] 
        upper_time_bound = retention_time + .05
        lower_time_bound = retention_time - .05
        if (float(exact_peak_retention) < float(upper_time_bound)) and (float(exact_peak_retention) > float(lower_time_bound)):
            peak_area = df.loc[peak_idx][data_headers["  Area  "]]
            print(f'Pass at: {retention_time}.')
        else:
            print(f'No peak found for retention time {retention_time}.')
            peak_area = 0.
        return peak_area, exact_peak_retention
    
    def map_fitness(self):
        """
        For each well, adds fitness values to the plate map and normalized them.
        """
        self.n_peakless_wells = 0
        for well in self.plate_map.keys():
            try:
                data_path = self.plate_map[well]['data_path']
                data = pd.read_csv(data_path,
                            sep=",",             
                            encoding="utf-16-le", 
                            header=None)
                
                # starting material
                print('\n',well)
                starting_material_area, _ = self.find_peak(self.starting_material, self.data_headers, data)
                self.plate_map[well]['starting_material_area'] = starting_material_area
                
                # internal standard
                internal_standard_area, _ = self.find_peak(self.internal_standard, self.data_headers, data)
                self.plate_map[well]['internal_standard_area'] = internal_standard_area
                
                # product
                product_area, _ = self.find_peak(self.product, self.data_headers, data)
                self.plate_map[well]['product_area'] = product_area

                # normalized product (to internal standard)
                product_standardized = self.plate_map[well]['product_area'] / (self.plate_map[well]['internal_standard_area'] + 1e-06)
                self.plate_map[well]['product_standardized'] = product_standardized
                
                # if any of the peaks are not found for a well, the well is set to zero:
                if any(x == 0 for x in [starting_material_area, internal_standard_area, product_area]):
                    self.n_peakless_wells += 1
                    self.plate_map[well]['starting_material_area'] = 0.
                    self.plate_map[well]['internal_standard_area'] = 0.
                    self.plate_map[well]['product_area'] = 0.
                    self.plate_map[well]['product_standardized'] = 0.
                

            except Exception as exc:
                print(f'Encountered exception in well {well}. Setting to zero: {exc}')
                self.plate_map[well]['starting_material_area'] = 0.
                self.plate_map[well]['internal_standard_area'] = 0.
                self.plate_map[well]['product_area'] = 0.
                self.plate_map[well]['product_standardized'] = 0.
                self.n_peakless_wells += 1
                
        # Normalize to the average of all parent wells
        self.parent_wells = ["C10", "C11", "C12", "G10", "G11", "G12"]
        avg_parent_product = np.mean([self.plate_map[parent_well]['product_standardized'] for parent_well in self.parent_wells])
        for well in self.plate_map.keys():
            try:
                # normalized product (to internal standard and parent wells)
                product_norm_to_parent = self.plate_map[well]['product_standardized'] / avg_parent_product
                self.plate_map[well]['product_norm_to_parent'] = product_norm_to_parent
            except Exception:
                print(f'Could not generate the normalization to parent wells for well {well}.')
        
        self.score = [float(self.plate_map[well]['product_norm_to_parent']) for well in self.plate_map.keys()]
        self.frac_improved = len([var for var in self.score if var > 1]) / len(self.score)
        out = {key: self.plate_map[key]['product_norm_to_parent'] for key in self.plate_map.keys()}
        for well, n in out.items():
            print(f"{well}:    {n:.2f}")
        
        ipdb.set_trace()



    def map_fitness_half(self):
        """
        Used when one plate is run under two conditions and hence requires separate
        normalization: 
        - Upper half (wells A01 - D12)
        - Lower hald (wells E01 - H12)
        For each well, adds fitness values to the plate map and normalized them.
        """
        self.n_peakless_wells = 0
        for well in self.plate_map.keys():
            try:
                data_path = self.plate_map[well]['data_path']
                data = pd.read_csv(data_path,
                            sep=",",             
                            encoding="utf-16-le", 
                            header=None)
                
                # starting material
                print('\n',well)
                self.well = well
                starting_material_area, _ = self.find_peak(self.starting_material, self.data_headers, data)
                self.plate_map[well]['starting_material_area'] = starting_material_area
                
                # internal standard
                internal_standard_area, _ = self.find_peak(self.internal_standard, self.data_headers, data)
                self.plate_map[well]['internal_standard_area'] = internal_standard_area
                
                # product
                product_area, _ = self.find_peak(self.product, self.data_headers, data)
                self.plate_map[well]['product_area'] = product_area

                # normalized product (to internal standard)
                product_standardized = self.plate_map[well]['product_area'] / (self.plate_map[well]['internal_standard_area'] + 1e-06)
                self.plate_map[well]['product_standardized'] = product_standardized
                
                # if any of the peaks are not found for a well, the well is set to zero:
                if any(x == 0 for x in [starting_material_area, internal_standard_area, product_area]):
                    self.n_peakless_wells += 1
                    self.plate_map[well]['starting_material_area'] = 0.
                    self.plate_map[well]['internal_standard_area'] = 0.
                    self.plate_map[well]['product_area'] = 0.
                    self.plate_map[well]['product_standardized'] = 0.

            except Exception as exc:
                print(f'Encountered exception in well {well}. Setting to zero: {exc}')
                self.plate_map[well]['starting_material_area'] = 0.
                self.plate_map[well]['internal_standard_area'] = 0.
                self.plate_map[well]['product_area'] = 0.
                self.plate_map[well]['product_standardized'] = 0.
                self.n_peakless_wells += 1
                
        # Normalize to the average of all parent wells
        rows_upper = ["A", "B", "C", "D"]
        rows_lower = ["E", "F", "G", "H"]
        wells_upper = []
        wells_lower = []

        well_number = 1
        for row in rows_upper:
            for col in range(1, 13):
                wells_upper.append(f"{row}{col:02d}") 
                well_number += 1
        well_number = 1
        for row in rows_lower:
            for col in range(1, 13):
                wells_lower.append(f"{row}{col:02d}") 
                well_number += 1
        
        self.parent_wells_upper = [] 
        self.parent_wells_lower = []

        for well in wells_upper:
            var = self.plate_map[well]['variant']
            if var == '#PARENT#':
                self.parent_wells_upper.append(well)
        for well in wells_lower:
            var = self.plate_map[well]['variant']
            if var == '#PARENT#':
                self.parent_wells_lower.append(well)

        avg_parent_product_upper = np.mean([self.plate_map[parent_well]['product_standardized'] for parent_well in self.parent_wells_upper])
        avg_parent_product_lower = np.mean([self.plate_map[parent_well]['product_standardized'] for parent_well in self.parent_wells_lower])
        
        for well in wells_upper:
            try:
                # normalized product (to internal standard and parent wells)
                product_norm_to_parent = self.plate_map[well]['product_standardized'] / avg_parent_product_upper
                self.plate_map[well]['product_norm_to_parent'] = product_norm_to_parent
            except Exception:
                print(f'Could not generate the normalization to parent wells for well {well}.')
        for well in wells_lower:
            try:
                # normalized product (to internal standard and parent wells)
                product_norm_to_parent = self.plate_map[well]['product_standardized'] / avg_parent_product_lower
                self.plate_map[well]['product_norm_to_parent'] = product_norm_to_parent
            except Exception:
                print(f'Could not generate the normalization to parent wells for well {well}.')
        
        self.score = [float(self.plate_map[well]['product_norm_to_parent']) for well in self.plate_map.keys()]
        self.frac_improved = len([var for var in self.score if var > 1]) / len(self.score) 
        self.score_unnormalized = [float(self.plate_map[well]['product_standardized']) for well in self.plate_map.keys()]#TODO remove   

    def well_format_converter(self, well: str):
        """
        Converts between well str formats: i.e. 'A01' -> 'A1' and vice versa.
        """
        row, num = well[0], well[1:]
        return f"{row}{int(num)}" if num[0] == "0" else f"{row}{int(num):02d}"

    def save_streamlit_partial(self):
        """
        Saves the levseq streamlit app partial in adequate format.
        """
        streamlit_data = []
        for i, well in enumerate(self.plate_map.keys()):
            try:
                data_path = self.plate_map[well]['data_path']
                data = pd.read_csv(data_path,
                            sep=",",             
                            encoding="utf-16-le", 
                            header=None)
                product_area, product_retention_time = self.find_peak(self.product, self.data_headers, data)
                row = [f'{i+1}', 'something-'+self.well_format_converter(well), 'something-'+self.well_format_converter(well), 'product', float(product_retention_time), float(product_area)]
                streamlit_data.append(row)
            except Exception as exc:
                print(f'Encountered and exception in creating streamlit partial for well: {well}, {exc}')
        headers = ['Sample Acq Order No', 'Sample Vial Number', 'Sample Name', 'Compound Name', 'RT [min]', 'Area']
        streamlit_df = pd.DataFrame(streamlit_data, columns=headers)
        out_path = os.path.join(self.out_dir, self.run_name+'.csv')
        streamlit_df.to_csv(out_path, index=False)
        print('Saved streamlit partial.')        

    def save_plate_array(self):
        """
        Saves the fitness and the mutations as a plate map (.csv) array.
        """
        try:
            fit_arr = pd.DataFrame([self.score[i:i+12] for i in range(0, 96, 12)])
            var_arr = pd.DataFrame([self.vars[i:i+12] for i in range(0, 96, 12)])
            plate = pd.concat([fit_arr, var_arr], axis=0).reset_index(drop=True)
            out_path = os.path.join(self.out_dir, self.run_name+'_plate.csv')
            plate.to_csv(out_path, header=False, index=False)
            print('Saved plate array.')
        except Exception as exc:
            print(f'Could not save plate array: {exc}')
    
    def save_plate_array_unnormalized(self):
        """
        Saves the fitness and the mutations as a plate map (.csv) array.
        """
        try:
            fit_arr = pd.DataFrame([self.score_unnormalized[i:i+12] for i in range(0, 96, 12)])
            var_arr = pd.DataFrame([self.vars[i:i+12] for i in range(0, 96, 12)])
            plate = pd.concat([fit_arr, var_arr], axis=0).reset_index(drop=True)
            out_path = os.path.join(self.out_dir, self.run_name+'_unnormalized_plate.csv')
            plate.to_csv(out_path, header=False, index=False)
            print('Saved plate array.')
        except Exception as exc:
            print(f'Could not save plate array: {exc}')


    def print_metrics(self):
        """
        Prints some metrics.
        """
        print(f'Number of wells, where at least one peak is missing: {self.n_peakless_wells}')
        print(f'Fraction of variants that are improved: {self.frac_improved:.2f}\n')
        print(f'Best variant fitness: {max(self.score):.2f}\n')
        print(f'Sorted fitness: {sorted(self.score, reverse=True)}\n')
        print(f'Unsorted fitness: {self.score}\n')
    
    def fit_to_ZS_table(self):
        """
        Adds the ground-truth fitness values to the ZS csv table of SSM.
        """
        ZS_table = pd.read_csv(self.ZS_table)
        
        for i, row in ZS_table.iterrows():
            var = row['var']
            var_activity = []
            for well in self.plate_map.keys():
                well_var = self.plate_map[well]['variant']
                if var == well_var:
                    var_activity.append(self.plate_map[well]['product_norm_to_parent'])
            try:
                if len(var_activity) != 0:
                    avg_var_activity = np.mean(var_activity)
                    ZS_table.loc[i, 'GT'] = avg_var_activity
            except:
                print(f'No GT data found for var {var}.')
            out_path = os.path.join(self.out_dir, 'ZS_correlation.csv')
            ZS_table.to_csv(out_path, index=False)
            print('Appended the GT values to ZS table.')
    
    
if __name__ == "__main__":
    def main():
        
        # argparsing
        parser = argparse.ArgumentParser()
        parser.add_argument("--run_json",
                    type=str,
                    help="Filepath to json with the run parameters")
        args = parser.parse_args()
        logfile_path = args.run_json

        # loading the json 
        with open(logfile_path, 'r') as f:
            runfile = json.load(f)

        # running the analysis
        evaluator = EvalBalrog(                 
                 balrog_basedir = runfile["balrog_basedir"],
                 run_name = runfile["run_name"],
                 parent_wells = runfile["parent_wells"],
                 out_dir = runfile["out_dir"],
                 starting_material = runfile["starting_material"],
                 internal_standard = runfile["internal_standard"],
                 product = runfile["product"],
                 visualization_partial = runfile["visualization_partial"],
                 ZS_table = runfile["ZS_table"]
                 )
        
        evaluator.seq_map()
        evaluator.map_fitness()
        evaluator.save_plate_array_unnormalized()
        evaluator.save_streamlit_partial()
        evaluator.print_metrics()
        evaluator.fit_to_ZS_table()

    main()
        
    
        


