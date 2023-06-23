# Maritime radar odometry

This repository contains code for estimating the motion of a ship, using only the navigation radar.


TODO: Add link to paper when published.

## Requirements

- Python 3.10 (GTSAM does not yet support python 3.11 on ubuntu 22.04)
- - venv (recommended, and used in the install instructions)
- opencv
- gtsam
- manif
- See `setup.py` for complete list of dependencies from pypi

## Install

```
# Download with --recurse-submodules to also get the dependencies.
git clone --recurse-submodules https://github.com/hflemmen/radar_odometry.git
# Optional but recomended to use a virtual environment
python3 -m venv venv
source venv/bin/activate
# Install dependencies
cd thirdparty/radar_utils
python3 -m pip install .
cd ../manif
python3 -m pip install .
cd ../..
python3 -m pip install .
```

## Example data

We have made a shortened, reduced resolution version of the sequence dubbed "b" in the paper openly available. You can the repository on that sequence by the following steps:

- Download the and unzip the data from [Zenodo](https://zenodo.org/record/8074028).
- Change the `in_path` parameter in `hardware_config.py` to the unziped folder. 
- Optional: Preprocess the images to save runtime as per the section below.
- Run `main.py` as described below.
## Executables

### Main
Configure the setup in both `software_config.py` and `hardware_config.py`. Then run

```commandline
python3 main.py
```

to test the algorithm.

The config files as given in the repo is for the associated test data. 


### Preprocessing

Subparts of the algorithm runs very slowly. Specifically, the sweep integration filter and the conversion to cartesian images are slow. 
Both of these can be precomputed by the standalone script "dataset_processing.py". Change the filepath to your dataset and run: 

```commandline
python3 dataset_processing.py $HOME/Data/b
```

To use the generated cache you need to set `use_cache=True` in `software_config.py`.


### Licence

This code is available under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

![by-nc-nd.eu.svg](files/by-nc-nd.eu.svg)