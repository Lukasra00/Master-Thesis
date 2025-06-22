# Master Thesis Data Generation Scripts

This repository contains the scripts used for the Master’s Thesis 
**\[Engineering Novel Enzyme Reactivities
with Zero-Shot Predictors]**.

## Repository Structure

* \`GCFID_post-processing`
  Scripts for the post-processing of variant fitness data from gas chromatogeaphy flame ionization detector.
* \`ZS`
  Contains all zero-shot predictors implemented in this work.   
* \`envs`
   Contains `.yml` files and `requirement.txt` files to run the scripts in this repo.
* \`simulated_annealing`
    Contains the scripts to run the implementation of parallel-tempering simulated annealing for protein optimization.
 
### Prerequisites
* **Python 3.9+** (tested on 3.10)
* **Conda** for environment management

### Installation
**Clone the repository**
   ```bash
   git clone https://github.com/Lukasra00/Master-Thesis.git
   cd Master-Thesis
   ```

## Usage

### Simulated Annealing Pipeline
Create and activate the simulated annealing venv by:
```bash
python -m venv SA_env
source SA_env/bin/activate
pip install -r envs/simulated_annealing_requirements.txt
```
The run parameters can be specified as in.
`simulated_annealing/LUT_example.json`

The run is started by:
```bash
cd simulated_annealing
python simulated_annealing/reSA.py --LUT_json path/to/LUT_example.json
```

### Zero-Shot Prediction Pipeline
Run preprocess_all() from substrate_aware.preprocess to preprocess the data.
Run individual zs predictors by:
```bash
cd ZS.zs
python example_zs_predictor.py
```


### GCFID Post-Processing
Run parameter are to be specified in the `run_GCFID_postprocessing.json` file.
```bash
cd GCFID_post-processing
python GCFID_Post-Processing.py --run_json run_GCFID_postprocessing.json
```



## License

[MIT License](LICENSE)

## Contact

Lukas Ra – [radtkel@ethz.ch](mailto:radtkel@ethz.ch)
Department of Biosystems Science and Engineering, 
ETH Zurich

Division of Chemistry and Chemical Engineering, 
California Institute of Technology

