# RASSP

## Summary

There are a few options for getting RASSP inference results on small molecules of your choice:
- Web API
    - We've setup a web API that runs FormulaNet/SubsetNet inferences on molecules at [spectroscopy.ai](https://www.spectroscopy.ai).
    - You can run inference on mols <= 48 atoms (for FormulaNet) and <= 64 atoms (for SubsetNet).
- Install RASSP locally.
- Build your own Docker image (TBD).
- Use our provided Docker image (TBD).

## Option 1. Local installation

First, clone this repo into the directory of your choice, e.g. `ROOTDIR=~/code/rassp-public`.

### 1. Local install of Anaconda environment

If you have Anaconda already installed, great.

If not, install Miniconda and Mamba like so:
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda init bash
```

Setup a new Conda environment using `rassp/environment.yml`:
- `cd rassp`
- `conda env create -q -n rassp -f environment.yml`
- `conda activate rassp`

### 2. Install `rassp` module
Install `rassp=1.0.0` as a local editable module (make sure to run it from the `rassp-public` root directory, where `setup.py` is located):
- `cd $ROOTDIR`
- `python -m pip install -e .`

### 3. Setup files and run script
Copy the expected files into their directories inside `rassp`:
- `rsync -razP models/ rassp/models/`
- `rsync -razP sample_data/ rassp/sample_data/`

Run the demo script that runs forward spectral prediction on a list of InChI strings inside `sample_data/in.txt`:
- `cd $ROOTDIR`
- Follow instructions in `rassp/run_rassp.py`

### DEBUGGING NUMPY VERSIONING ISSUES
- Depending on how your Anaconda installation resolved the installations, you may get the following issues:
    - Numba version incompatibility `ImportError: Numba needs NumPy 1.21 or less`
    - Tinygraph (JonasLab library) incompatibility `ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject`
- The Numba error tells us that we should install `numpy<=1.21`
- The Tinygraph error is cryptic, but it has to do with a change in the Numpy API at version 1.20.0
- To resolve this, uninstalling Numpy and reinstalling it with a specific version should fix things:
    - `pip uninstall numpy`
    - `pip install numpy==1.21`
  
### Option 2. Compile your own Docker image (TBD)

TBD.

### Option 3. Use our pre-compiled Docker image (TBD)

TBD.

## Codebase guide

`rassp` module:
- `expconfig`: YAML config files for specifying experiments and models
- `msutil`: Fast code for computing mass spectra and molecule subsets
- `model`: Model logic
- `featurize`: Molecule featurization
- `dataset`: Dataset object
- `datagen`: Datagen scripts
- `docker`: Running a Docker image to run inference using a given model checkpoint
- `util.py`: Generic utils
- `netutil.py`: Model utils
- `forward_evaluate_pipeline.py`: Batch forward inference script
- `forward_train.py`: Main training script
- `run_rassp.py`: Inference script for running a pre-trained model against molecules
- `metrics.py`: Metrics functions, including SDP, DP, and others

`rassp-public` module:
- `library_match_pipeline.py`: Library matching / database lookup metrics
- `analysis_pipeline.py`: Forward model metrics
- `const.py`: Configures analysis scripts
- `sample_data`: Parquet files containing sample datasets to train and eval against
    - Should be copied into `rassp` folder
- `models`: Pretrained model weights and checkpoints
    - Should be copied into `rassp` folder

Training, inference, and analysis artifacts (will be generated upon running scripts)
- `checkpoints`: PyTorch model checkpoints
- `tblogs.formulae`: Tensorboard logs
- `forward.preds`: Forward inference results
- `results.metrics`: Metrics of forward inference
- `library_match_results.metrics.mass_filter_15.reg_dp`: Library matching metrics

## Pretrained model weights

Our pretrained SubsetNet and FormulaNet model weights can be found in `rassp-public/models`.

All model weights and files can also be located here (TBD): [https://people.cs.uchicago.edu/~ericj/rassp/](https://people.cs.uchicago.edu/~ericj/rassp/)

Pretrained model weights (both the `.model` and `.meta` files) should be downloaded to `rassp/models` for our scripts to work.

## Provided dataset

We take the first 100 molecules from the `smallmols` dataset [1].
We then run `cfm-predict` against them and save their spectra in `sample_data/smallmols_cfm_pred_public_sample.parquet`.
We split this 100 mol dataset into 2x 50 mol datasets saved as `sample_data/smallmols_cfm_pred_public_sample.0.parquet`
and `sample_data/smallmols_cfm_pred_public_sample.1.parquet`. These two non-overlapping datasets are used in the library matching pipeline later on.

The columns:
- `mol_id`: String
  - Generally can be String or Int, but `smallmols` labels them with strings indexing them against the NIST 2014 database they were pulled from.
- `inchi`: String
- `inchi_key`: String
  - Hash of the `inchi` string.
  - Generated from `inchi` via `Chem.InchiToInchiKey(inchi)`.
- `smiles`: String
- `rdmol`: LargeBinary
  - A binary blob. To get RDKit molecules from a `rdmol` binary, we need to do `Chem.Mol(rdmol)`.
- `cv_id`: Int
  - Cross-val split index. Used to subdivide data into train and test sets.
- `morgan4_crc32`: Int
  - CRC32 checksum of the `morgan4` fingerprint of molecule, used to compute the `cv_id`.
- `spect`: List[Tuple[Float, Float]]
  - A spectrum is represented as a list of 2-tuple (mass, intensity) pairs.

## Training a demo model

Run the following example command from within the `eimspred_public` repo:
`USE_CUDA=1 CUDA_VISIBLE_DEVICES='<GPU_ID>' python rassp/forward_train.py rassp/expconfig/demo.yaml first-test`

Change the `GPU_ID` to an integer 0, 1, etc if you have multiple GPUs, or an empty string if you are not running on GPU.

Output:
- `checkpoints` - Location of model checkpoints
    - `<yaml_basename>.<additional_name>.<timestamp>.<epoch>.model` - path pattern
    - `demo.first-test.48668593.00000000.model` - example path
    - `demo.first-test.48668593.00000000.state` - example path
- `tblogs.formulae` - Location of Tensorboard intermediate results

## Forward evaluation of demo model on a molecular dataset

Grab the model name from the checkpoint directory `checkpoints`. For example, our model name might look like: `demo.first-test.48668593`.
In `const.py`, we'll add a new entry in `FORWARD_EVAL_EXPERIMENTS`:
```
FORWARD_EVAL_EXPERIMENTS = {
  'demo': {
    'dataset' : './sample_data/smallmols_cfm_pred_public_sample.parquet',
    'cv_method' : {
      'how': 'morgan_fingerprint_mod', 
      'mod' : 10,
      'test': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    'normalize_pred': True,
    'streaming_save': True,
    'checkpoint': 'checkpoints/demo.first-test.48668593',
    'batch_size': 6,
    'epoch': 0,
    'mol_id_type': str,  # either str or int, depending on your input dataset's `mol_id` column dtype
  },
}
```

To execute forward evaluation against all the experiments in `FORWARD_EVAL_EXPERIMENTS`, we'll run:
`USE_CUDA=1 CUDA_VISIBLE_DEVICES='<GPU_ID>' python rassp/forward_evaluate_pipeline.py`

In the example experiment `demo` we have provided, the output spectra will be saved to a `.sqlite` file at
`forward.preds/demo.spect.sqlite`.

The columns:
- `mol_id`: String | Int
  - Index labeling the molecule. Can be either a string or an integer, depending on the column in input dataset.
    We assume integer by default, otherwise it needs to be explicitly specified as `mol_id_type` in `const.py`.
- `spect`: LargeBinary
  - Pickled list. When unpickled, we get `List[Tuple[Float, Float]]`, where the inner 2-tuple comprises (mass, intensity) pairs.
- `phase`: String
  - Either 'train' or 'test'. If running inference on all molecules, we set all phases to 'test' by
    putting all splits in the `cv_method` dictionary in `const.py`.

## Analysis and metrics of forward model
Edit `const.py` `ANALYSIS_EXPERIMENTS` with the `pred_spect` path pointing to the output from `forward_evaluate_pipeline.py`

Run and get metrics for the model by running:
`python analysis_pipeline.py`

Output goes into `results.metrics`.

## Library matching metrics
Edit `const.py` `LIBRARY_MATCH_EXPERIMENTS` with the `exp_name` set to the key for the experiment you want to run library matching metrics on in the `ANALYSIS_EXPERIMENTS`.

Notes:
- Unlike previous pipelines, this pipeline assumes the existence of the `inchi_key` column in the `main_library` and `query_library` Parquet files.
- The main library and query library molecules must be strictly non-overlapping. There is an assert to check for this.

Run and get metrics for the model by running:
`python library_match_pipeline.py`

Output goes into `library_match_results.metrics.mass_filter_<MASS_FILTER_DELTA>.<DP_NAME>`.

## Using `run_rassp.py` to predict spectra for your own molecules

Example usage:
- Ensure that you are in the `rassp` directory.
- Copy the `models` directory to `rassp/models`.
- Copy the `sample_data` directory to `rassp/sample_data`.
- Write your molecules as smiles / inchi strings to `sample_data/in.txt`
- Run `run_rassp.py`, using the instructions provided in the script.
- Spectra are stored in `sample_data/out.txt` (or whatever `output_filename` path you specified.)

## All the steps, in order

CPU train and eval:
```
# install conda environment per instructions
# install rassp as local package
pip install -e .

# train a model
USE_CUDA=0 python rassp/forward_train.py rassp/expconfig/demo.yaml first-test

# run model against mols to get predicted spectra
# edit const.py to point to the right model checkpoint, eg `checkpoints/demo.first-test.48755607`
USE_CUDA=0 python rassp/forward_evaluate_pipeline.py

# compute forward spectral metrics
python analysis_pipeline.py

# compute library matching metrics
python library_match_pipeline.py
```

## Using GPUs

If a GPU is available, PyTorch GPU will attempt to use it.
If multiple GPUs are available, you should explicitly specify the index of the device that you want to use by prepending `CUDA_VISIBLE_DEVICES="<index:int>"` to your python command.
Multi-GPU training is possible but finicky.
We recommend not doing so unless you're quite familiar with distributed GPU training.

If a GPU is not available and/or Nvidia drivers are not available, you will need to train with `USE_CUDA=False` inside `rassp/forward_train.py`, and add an environment flag 
prior to executing each python script, eg:
```
USE_CUDA=0 CUDA_VISIBLE_DEVICES="" python rassp/forward_train.py rassp/expconfig/demo.yaml first-test
USE_CUDA=0 CUDA_VISIBLE_DEVICES="" python rassp/forward_evaluate_pipeline.py
```

## References
1. RASSP. 2023. URL: https://spectroscopy.ai/papers/rassp/
2. CFM-ID. 2021. URL: https://cfmid.wishartlab.com/
