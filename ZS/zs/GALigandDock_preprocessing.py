# Script to preprocess the input files for GALigandDock
# Authors: Lukas Radtke {lradtke@caltech.edu}
# Date: 16/03/25

import subprocess
import os
import ipdb
import pymol2
import ipdb.stdout
import argparse


from glob import glob
from copy import deepcopy
from tqdm import tqdm
from typing import Union
from Bio.PDB import PDBParser, Superimposer, PDBIO

from  Master_Thesis.ZS.preprocess import ZSData

parser = argparse.ArgumentParser(
                    prog='Pyrosetta Pipeline',
                    description='This program prepares the inputs for PyRosetta GALigandDock. Type "-h" for help.')
parser.add_argument("--meta_string", 
                    type=str, 
                    help="Path to campaign metadata csv-file."
                    )
parser.add_argument("--meta_list", 
                    nargs='+', 
                    help="List of paths to campaign metadata csv-files spell out paths separated by a space."
                    )
parser.add_argument("--struc_dir", 
                    type=str, 
                    help="Path to directory with docked input PDB files of campaign."
                    )
parser.add_argument("--tmp_dir", 
                    type=str, 
                    help="Path to directory where temporary files can be stored."
                    )
parser.add_argument("--out_dir", 
                    type=str, 
                    help="Path to directory where preprocessed output files should be stored."
                    )
parser.add_argument("--rosettascript_path", 
                    type=str, 
                    help="Path mol2genparams.py (\nHint: Script is part of your local pyrosetta installation and often found under rosetta/source/scripts/python/public/generic_potential/mol2genparams.py)."
                    )
parser.add_argument("--net_charge_unit_1", 
                    type=int, 
                    help="Contains formal charges of the ligand unit 1 (Unit with less atoms first.)"
                    )
parser.add_argument("--net_charge_unit_2", 
                    type=int, 
                    help="Contains formal charges of the ligand unit 2 (Unit with less atoms first.)"
                    )
args = parser.parse_args()
if args.meta_string is not None:
    pattern = args.meta_string
elif args.meta_list is not None:
    pattern = args.meta_list
else:
    print(f'ERROR: No valid campaign meta csv path provided.')
kwargs = {"structure_dir": args.struc_dir,
          "preprocessing_dir": args.tmp_dir,
          "preprocessed_dir": args.out_dir,
          "rosetta_param_script_path": args.rosettascript_path,
          "charge_dict": {'unit1': args.net_charge_unit_1, 'unit2': args.net_charge_unit_2}}


class PyrosettaData(ZSData):

    def __init__(
        self,
        input_csv: str,
        structure_dir: str = "data/structure",
        preprocessing_dir: str ="data/tmp",
        preprocessed_dir: str ="data/preprocessed",
        rosetta_param_script_path: str = "./rosetta/source/scripts/python/public/generic_potential/mol2genparams.py",
        charge_dict: dict = {'unit1': 0, 'unit2':1} # describes overall formal charge of each unit (unit 1 is always smaller) 
    ):
        super().__init__(            
            input_csv=input_csv,
            structure_dir=structure_dir, 
            )
        
        self.rosetta_param_script_path = rosetta_param_script_path
        self.structure_dir = structure_dir
        self.preprocessing_dir = preprocessing_dir
        self.preprocessed_dir = preprocessed_dir
        self.charge_dict = charge_dict
        self.campaign = os.path.split(input_csv)[-1][:-4]

        self.preprocess_inputs()
    
    
    def scrape_and_convert(self):
        """
        Collects all .cif files from a campaign directory and converts them to pdb in the preprocessing dir. 
        Expects .cif files to be in structure self.structure dir > campaign_name > variant_name> variant_name_model.cif
        """
        campaign_names = os.listdir(self.structure_dir)
        capaign_out_dir = os.path.join(self.preprocessing_dir, self.campaign)
        os.makedirs(capaign_out_dir, exist_ok=True)
        self.clean_tmp(capaign_out_dir, [' '])
        print(f'INFO: Scraping and converting to PDB for {self.campaign}.')
        for variant_name in os.listdir(os.path.join(self.structure_dir, self.campaign)):
            variant_path = os.path.join(self.structure_dir, self.campaign, variant_name, variant_name+'_model.cif')
            out_path = os.path.join(capaign_out_dir, variant_name+'.pdb')
            
            
            cli = ['obabel',
                variant_path,
                '-O',
                out_path    
            ]
            cli_return = subprocess.run(cli, capture_output=True)
    

    def convert_pdb_name(self):  
        campaign_dir = os.path.join(self.preprocessing_dir, self.campaign)
        pattern_1 = os.path.join(campaign_dir, f"*.pdb")
        for file_path in glob(pattern_1):
            os.remove(file_path)
        
        cli = ['cp', 
               '-r', 
               '/disk2/fli/REVIVAL2/zs/hydro/md_mut/'+self.campaign, 
               '/disk2/lukas/EnzymeOracle/data/md_mut_immutable/',
                ]
        subprocess.run(cli)

        immutable_dir = '/disk2/lukas/EnzymeOracle/data/md_mut_immutable'
        immutable_campaign = os.path.join(immutable_dir, self.campaign)
        for variant in os.listdir(immutable_campaign):
            variant_path_old = os.path.join(immutable_campaign, variant)
            basename = os.path.split(variant_path_old)[1]
            basename = os.path.basename(variant_path_old)
            basename = basename.replace(':', '_').lower()
            out_path = os.path.join(self.preprocessing_dir, self.campaign, basename)
            os.rename(variant_path_old, out_path)


    def check_unusual_ligand_atoms(self):
        for enzyme in tqdm(os.listdir(os.path.join(self.preprocessing_dir, self.campaign)), desc='INFO: Checking for am1bcc incompatible atoms in your campaign.'):
            enzyme_sample_path = os.path.join(self.preprocessing_dir, self.campaign, enzyme)
            with open(enzyme_sample_path, 'r') as PDB:
                sample_pdb_lines = PDB.readlines()
            
            allowed_elements = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}
            atom_type_list = []
            for line in sample_pdb_lines:
                if line.startswith(('ATOM', 'HETATM', 'LIG', 'UNL', 'UNK')):
                    element = line[76:80].strip()
                    atom_type_list.append(element)
            if not set(atom_type_list).issubset(allowed_elements):
                print(f'INFO: Unusual element(s) {set(atom_type_list) - allowed_elements} detected in your enzyme {os.path.split(enzyme_sample_path)[-1]}.')
                print(f'INFO: am1bcc charge model only supports {allowed_elements}. ')
                print(f'INFO: Your ligand will be removed and saved as mol2 for you to manually replace the problematic atom(s).')
                unusual_elements_flag = True
                break
            else:
                unusual_elements_flag = False
                break
        return unusual_elements_flag
            
    
    def superimpose(self, ref_path):
        parser = PDBParser(QUIET=True)
        super_imposer = Superimposer()

        variant_list = os.listdir(os.path.join(self.preprocessing_dir, self.campaign))
        ref_struc = parser.get_structure("ref", ref_path)
        mobile_structures_paths = [os.path.join(self.preprocessing_dir, self.campaign, var) for var in variant_list[1:] if not var.endswith('ligand.pdb') and not var.endswith('ligand.mol2')] 

        for mobile_struc_path in tqdm(mobile_structures_paths, desc=f'INFO: Aligning variant positions for campaign {self.campaign}'):
            mobile_struc = parser.get_structure('mobile', mobile_struc_path)

            ref_atoms = [atom for atom in ref_struc.get_atoms() if atom.get_id() == "CA"]
            mobile_atoms = [atom for atom in mobile_struc.get_atoms() if (atom.get_id() == "CA" and atom.get_parent().get_id()[0] == ' ' and atom.get_parent().get_resname() in ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"))]

            super_imposer.set_atoms(ref_atoms, mobile_atoms)
            super_imposer.apply(mobile_struc.get_atoms())

            aligned_struc_path = mobile_struc_path[:-4]+'_aligned.pdb'
            io = PDBIO()
            with open(aligned_struc_path, "w") as aligned_file:
                io.set_structure(mobile_struc)
                io.save(aligned_file)
            if not os.path.exists(aligned_struc_path):
                print(f'ERROR: The file {aligned_struc_path} could not be saved.')
        """
        ref_path = ref_path[:-4]+'_aligned.pdb'
        with open(ref_path, 'w') as ref_file:
            io.set_structure(ref_struc)
            io.save(ref_file)
        """
        if not os.path.exists(ref_path):
            print(f'ERROR: The file {ref_path} could not be saved.')




    def remove_ligand(self, ref_path):
        # splitting enzyme and ligand 
        aligned_pdbs = [f for f in os.listdir(os.path.join(self.preprocessing_dir, self.campaign)) if f.endswith('_aligned.pdb')]
        for aligned_pdb in aligned_pdbs:
            aligned_struc_path = os.path.join(self.preprocessing_dir, self.campaign, aligned_pdb)
            with open(aligned_struc_path, 'r') as aligned_file:
                pdb_lines = aligned_file.readlines()
                
                enzyme_lines = []
                for line in pdb_lines:
                    if line.startswith(('ATOM', 'HETATM')):
                        res_name = line[17:20].strip()
                        alt_name = line[12:15].strip()
                        if res_name not in ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE","LEU", "LYS", "MET", "PHE", "PRO","SER", "THR", "TRP", "TYR", "VAL") and alt_name not in ("ALA", "ARG", "ASN", "ASP", "CYS","GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"):
                            pass
                        else:
                            enzyme_lines.append(line)

                enzyme_pdb_file = os.path.basename(aligned_struc_path)[:-4] + '_enzyme' + '.pdb'
                enzyme_pdb_filepath = os.path.join(self.preprocessing_dir, self.campaign, enzyme_pdb_file)
                    
                with open(enzyme_pdb_filepath, 'w') as enzpdb:
                    enzpdb.writelines(enzyme_lines)
            
            with open(ref_path, 'r') as ref:
                ref_pdb_lines = ref.readlines()
                ligand_lines = []
                for line in ref_pdb_lines:
                    if line.startswith(('ATOM', 'HETATM')):
                        res_name = line[17:20].strip()
                        alt_name = line[12:15].strip()
                        if res_name not in ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HOH") and alt_name not in ("ALA", "ARG", "ASN", "ASP", "CYS","GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HOH"):
                            ligand_lines.append(line)
                        else:
                            pass

        ligand_pdb_file = self.campaign + '_ligand' + '.pdb'
        ligand_pdb_filepath = os.path.join(self.preprocessing_dir, self.campaign, ligand_pdb_file)
            
        with open(ligand_pdb_filepath, 'w') as ligpdb:
            ligpdb.writelines(ligand_lines)
        
        ligand_mol2_filepath = ligand_pdb_filepath[:-4] + '.mol2'
        cli = ['obabel',
               ligand_pdb_filepath,
               '-O',
               ligand_mol2_filepath
                ]
        subprocess.run(cli)
        print('INFO: Saved a sample ligand as .mol2 for you to manually replace unusal atoms.')
        print(f'INFO: Saved sample ligand at {ligand_mol2_filepath}.')

        
    def insert_corrected_ligand(self):
        # merging enzyme and ligand again
        aligned_pdbs = [f for f in os.listdir(os.path.join(self.preprocessing_dir, self.campaign)) if f.endswith('_aligned_enzyme.pdb')]
        corrected_ligand_path = os.path.join(self.preprocessed_dir, self.campaign, self.campaign+'_ligand_corrected.mol2')
        print(f'INPUT REQUIRED: Please manually inspect your ligand in a molecular viewer GUI application.')
        print(f'INPUT REQUIRED: Correct any unusual atoms in your ligand and add Hydrogens, also inspect correctness of bonds.')
        print(f'INPUT REQUIRED: Save corrected ligand under {corrected_ligand_path} and input "c" into the console, then press ENTER.')
        ipdb.set_trace()
        # LOOK AT CONSOLE ABOVE
        for aligned_pdb in tqdm(aligned_pdbs, desc=f'INFO: Inserting corrected ligand back into enzymes.'):

            empty_enzyme_path = os.path.join(self.preprocessing_dir, self.campaign, aligned_pdb)
            with pymol2.PyMOL() as pymol:
                cmd = pymol.cmd
                
                cmd.load(empty_enzyme_path, 'emptyenzyme')
                cmd.load(corrected_ligand_path, 'ligand')
                
                out_path = os.path.join(self.preprocessing_dir, self.campaign, aligned_pdb[:-11]+'_merged.pdb')
                cmd.save(out_path, "all")
    

    def clean_tmp(self, dir, file_endings_to_keep):
        file_whitelist = []
        for file_ending in file_endings_to_keep:
            file_whitelist += [f for f in os.listdir(dir) if f.endswith(file_ending)]
        for file in os.listdir(dir):
            filepath = os.path.join(dir, file)
            if file not in file_whitelist:
                os.remove(filepath)
        print(f'INFO: Cleaned tmp files not ending with "{file_endings_to_keep}" from {dir}')


    def split_enzyme_and_ligand(self, pdb_lines, struc_path):
        ligand_lines = []
        enzyme_lines = []


        for line in pdb_lines:
            if line.startswith(('ATOM', 'HETATM')):
                res_name = line[17:20].strip()
                alt_name = line[12:15].strip()
                if res_name in ('LIG', 'UNL', 'UNK') or alt_name in ('LIG', 'UNL', 'UNK'):
                    ligand_lines.append(line)
                else:
                    enzyme_lines.append(line)

        ligand_pdb_file = os.path.basename(struc_path)[:-11] + '_ligand' + '.pdb'
        ligand_pdb_filepath = os.path.join(self.preprocessing_dir, self.campaign, ligand_pdb_file)

        enzyme_pdb_file = os.path.basename(struc_path)[:-11] + '_enzyme' + '.pdb'
        enzyme_pdb_filepath = os.path.join(self.preprocessed_dir, self.campaign, enzyme_pdb_file)

        with open(ligand_pdb_filepath, 'w') as ligpdb:
            ligpdb.writelines(ligand_lines)
            print(f'INFO: Ligand atoms succesfully extracted and saved as .pdb file at {ligand_pdb_file}.')
        
        with open(enzyme_pdb_filepath, 'w') as enzpdb:
            enzpdb.writelines(enzyme_lines)
            print(f'INFO: Enzyme atoms succesfully extracted and saved as .pdb file at {enzyme_pdb_file}.')
        
        return enzyme_pdb_filepath, ligand_pdb_filepath
    
    
    def pdb2mol2(self, ligand_pdb_path):
        out_path = os.path.join(self.preprocessing_dir, os.path.basename(ligand_pdb_path))[:-4] + '.mol2'
        cli = ['obabel',
               ligand_pdb_path,
               '-O',
               out_path]
        cli_return = subprocess.run(cli, capture_output=True, text=True)
        print(f'INFO: Succesfully generated .mol2 file at {out_path}')
        return out_path


    def generate_charges(self):
        # if multiple units are present, they must be split in separate mol2 for antechamber
        corrected_ligand_path = os.path.join(self.preprocessed_dir, self.campaign, self.campaign+'_ligand_corrected.mol2')
        out_path = os.path.join(self.preprocessed_dir, self.campaign, os.path.split(corrected_ligand_path)[-1][:-5]+'_unit.mol2')
        cli = ['obabel',
                corrected_ligand_path,
                '-O',
                out_path,
                '-m',
                '--separate'       
            ]
        cli_return = subprocess.run(cli, capture_output=True, text=True)
        unit_path_list = [out_path[:-5]+'1.mol2', out_path[:-5]+'2.mol2']
        if os.path.exists(unit_path_list[0]) and  os.path.exists(unit_path_list[1]):
            print(f'INFO: Succesfully split ligand mol2 file along its units.')
        else:
            print(f'ERROR: Splitting of mol2 file along its units failed')
        
        # generate the am1bcc charges for both units separately using antechamber
        print(f'INFO: Generating am1bcc charges. This may take a moment.')
        
        charged_unit_paths = []
        for unit_path in unit_path_list: 
            unit_key = unit_path[-10:-5]
            os.makedirs(os.path.join(self.preprocessed_dir, self.campaign), exist_ok=True)
            out_path = os.path.join(self.preprocessed_dir, self.campaign, os.path.split(unit_path)[-1][:-5]+'_am1bcc.mol2')
            if not os.path.exists(unit_path):
                print('ERROR: No unit.mol2 file found')
            cli = ['antechamber',
                '-i',
                unit_path,
                '-fi',
                'mol2',
                '-o',
                out_path,
                '-fo', 
                'mol2',
                '-c', 
                'bcc',       # defines chargemethod bcc = am1bcc
                '-nc',       # defines net charge
                str(self.charge_dict[unit_key]),
                '-dr',       # ignores frozen bonds
                'no',
                '-at',
                'sybyl'
            ]
            cli_return = subprocess.run(cli, capture_output=True, text=True)
            charged_unit_paths.append(out_path)
            if os.path.exists(out_path):
                print(f'INFO: Sucessfully generated am1bcc charges for {out_path}.')
            else:
                print(f'ERROR: Charge generation failed for {out_path}.')

        # merging the two units back together 
        out_path = unit_path[:-11]+'recombined_units_am1bcc.mol2'
        cli = ['obabel'] + charged_unit_paths + [
               '-O',
               out_path, 
               '--combine', 
               'm']
        cli_return = subprocess.run(cli, capture_output=True, text=True)
        if os.path.exists(out_path):
            print(f'INFO: Succesfully recombined units into a am1bcc_charged.mol2 file.')
        else:
            print(f'ERROR: Recombination of units into a am1bcc_charged.mol2 file failed.')

        return charged_unit_paths
        

    def generate_params(self, charged_unit_paths):
        params_paths = []
        for charged_unit_path in charged_unit_paths: 
            out_dir = os.path.join(self.preprocessed_dir, self.campaign)
            # generate params file
            cli = ['python',
                self.rosetta_param_script_path,
                '-s',
                charged_unit_path,
                '--outdir',
                out_dir,
                '--no_pdb'
            ]
            cli_return = subprocess.run(cli, capture_output=True, text=True)

            
            # rename the internal name of the params file
            unit_num = charged_unit_path[-13:-12].strip()
            new_paramsfile_lines = []
            corrected_params_path = os.path.join(out_dir, charged_unit_path[:-5]+'.params')
            chain_names =['X', 'Y', 'Z']
            with open(corrected_params_path, 'r') as paramsfile:
                paramsfile_lines = paramsfile.readlines()
            for paramsfile_line in paramsfile_lines:
                if paramsfile_line.startswith('NAME'):
                    new_paramsfile_lines.append(paramsfile_line[:-2]+unit_num+'\n')
                elif paramsfile_line.startswith('IO_STRING'):
                    new_paramsfile_lines.append(paramsfile_line[:-4]+unit_num+f' {chain_names[int(unit_num)-1]}\n')
                else:
                    new_paramsfile_lines.append(paramsfile_line)
            final_params_path = os.path.join(self.preprocessed_dir, self.campaign, os.path.split(charged_unit_path)[-1][:-5]+'.params')
            with open(final_params_path, 'w') as paramsfile:
                paramsfile.writelines(new_paramsfile_lines)
            params_paths.append(final_params_path)
        return params_paths



    def ligands_into_enzyme(self, unit_list, enzyme_pdb_path):
        # convert the ligand_am1bcc.mol2 to .pdb
        ligand_paths = []
        for unit_path in unit_list:
            cli = ['obabel',
                    unit_path,
                    '-O', 
                    unit_path[:-5]+'.pdb'
            ]
            cli_return = subprocess.run(cli, capture_output=True, text=True)
            ligand_paths.append(unit_path[:-5]+'.pdb')

        # find last indeces on the enzyme pdb
        with open(enzyme_pdb_path, 'r') as enzyme_pdb:
            enzyme_lines = enzyme_pdb.readlines()
        for enzyme_line in enzyme_lines:
            if enzyme_line.startswith('ATOM' or 'HETATM'):
                atom_num = int(enzyme_line[5:12].strip())
                residue_num = int(enzyme_line[22:27].strip())
        enzyme_lines.append('TER \n')


        # reformat the ligand lines accordingly and add them to the enzyme
        for ligand_path in ligand_paths:
            residue_num += 1
            residue_name = 'LG'+ligand_path[-12:-11].strip()
            chain_names = ['X', 'Y', 'Z']

            with open(ligand_path, 'r') as ligand_pdb:
                ligand_lines = ligand_pdb.readlines()
            
            for ligand_line in ligand_lines:
                if ligand_line.startswith('ATOM') or ligand_line.startswith('HETATM'):
                    res_name = ligand_line[17:20].strip()
                    alt_name = ligand_line[12:15].strip()
                    #if res_name in ('LIG', 'UNL', 'UNK') or alt_name in ('LIG', 'UNL', 'UNK'):
                    ligand_line_list = ligand_line.split()
                    atom_num += 1
                    try:
                        new_line = "{:<6}{:>5}  {:<4}{:>3} {:>1}{:>4}{:>11}{:>8}{:>8} {:>6}{:>6}{:>12}\n".format(
                                        'HETATM',  # Record type
                                        atom_num,             # Atom serial number
                                        ligand_line_list[2],  # Atom name
                                        residue_name,         # Residue name
                                        chain_names[int(residue_name[-1:])-1],       # Chain identifier
                                        residue_num,          # Residue sequence number
                                        ligand_line_list[6],  # X coordinate
                                        ligand_line_list[7],  # Y coordinate
                                        ligand_line_list[8],  # Z coordinate
                                        ligand_line_list[9],  # Occupancy
                                        ligand_line_list[10], # Temperature factor
                                        ligand_line_list[11]  # Element symbol
                        )
                    except (ValueError, IndexError):
                        ipdb.set_trace()
                    
                    enzyme_lines.append(new_line) 
        enzyme_lines.append('TER \n')   
        enzyme_string = ''.join(enzyme_lines)
        save_path = os.path.join(self.preprocessed_dir, self.campaign, os.path.basename(enzyme_pdb_path)[:-4]+'_final.pdb')
        with open(save_path, 'w') as outpath:
            outpath.writelines(enzyme_string)
    
        
    def preprocess_inputs(self):
        """
        Input .pdb file is expected to hold an enzyme with docked ligand.
        Ligand requires correct bond orders and hydrogens to be defined. 
        The active site must not contain single atoms (e.g. ions), or molecules (unit) with less than four atoms.
        """
        csv_path = self._input_csv
        
        os.makedirs(os.path.join(self.preprocessed_dir,), exist_ok=True)
        print(f'INFO: Starting pipeline for campaign {self.campaign}.')

        # Scrapes all .cif files (AF3 output folder format) and converts them to pdb
        #self.scrape_and_convert()

        # Checks for elements for which no am1bcc charges can be generated 
        # Only 'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I' are allowed.
        ref_path = os.path.join('/disk2/lukas/EnzymeOracle/data/md_mut_ref', self.campaign+".pdb")
        self.convert_pdb_name()

        if self.check_unusual_ligand_atoms():

            self.convert_pdb_name()

            # superimpose all pdb structures spatially
            self.superimpose(ref_path)
            
            # remove ligand 
            self.remove_ligand(ref_path)
            
            # insert corrected ligand (refer to methods of how to manually correct the ligand here)
            self.insert_corrected_ligand()
        else:
            # extract ligand as mol2 for charge generation
            self.remove_ligand()

        # clean up temp dir and only keep the enzymes with inserted corrected ligand
        self.clean_tmp(os.path.join(self.preprocessing_dir, self.campaign), ['_aligned_merged.pdb', '.mol2'])

        # generate am1bcc charges for ligand.mol using antechamber
        charged_unit_list = self.generate_charges()

        # generate params file from ligand.mol2
        params_paths = self.generate_params(charged_unit_list)

        # merge ligand.pdb with protein.pdb
        for enzyme_name in tqdm(os.listdir(os.path.join(self.preprocessing_dir, self.campaign)), desc=f'INFO: Merging enzyme with charged'):
            if enzyme_name.startswith(self.campaign+'_ligand') or enzyme_name.startswith(self.campaign+'__enzyme'):
                pass
            else:
                enzyme_pdb_path = os.path.join(self.preprocessing_dir, self.campaign, enzyme_name)
                with open(enzyme_pdb_path, 'r') as PDB:
                    enzyme_lines = PDB.readlines()
                empty_enzyme, _ = self.split_enzyme_and_ligand(enzyme_lines, enzyme_pdb_path)
                self.ligands_into_enzyme(charged_unit_list, empty_enzyme)


def run_pyrosetta_pipeline(pattern: Union[str, list] = None, kwargs: dict = {}):
    if isinstance(pattern, str):
        lib_list = glob(pattern)
    else:
        lib_list = deepcopy(pattern)

    for lib in lib_list:
        print(f"Running Pyrosetta Pipeline for {lib}...")
        PyrosettaData(input_csv=lib, **kwargs)

