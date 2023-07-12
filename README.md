# RETSINA: Reproducibility and Experimentation Testbed for Signal-Strength Indoor Near Analysis

This repository holds the code for the RETSINA project as published and presented at IPIN2023.

## Dependencies

This library was implemented using Python 3.10 and a virtual environment with the following dependencies:

- certifi==2022.12.7
- charset-normalizer==3.0.1
- contourpy==1.0.7
- cPython==0.0.6
- cycler==0.11.0
- Cython==0.29.33
- dask==2023.5.0
- dnspython==2.3.0
- fonttools==4.38.0
- idna==3.4
- imbalanced-learn==0.10.1
- joblib==1.2.0
- kiwisolver==1.4.4
- matplotlib==3.7.0
- numpy==1.24.1
- packaging==23.0
- pandas==1.5.3
- Pillow==9.4.0
- pyparsing==3.0.9
- python-dateutil==2.8.2
- pytz==2022.7.1
- requests==2.28.2
- scikit-learn==1.2.1
- scikit-plot==0.3.7
- scipy==1.10.0
- six==1.16.0
- threadpoolctl==3.1.0
- urllib3==1.26.14
- pymrmr==0.1.11 **Note that this dependency is not in the requirements.txt file and requires being installed after
  preinstalling other dependencies such as Cython via the pip install command**

These dependencies are also provided in requirements.txt found in the repository's root directory.

## Prerequisites

- A Python development environment with the dependencies as specified in the previous section.
- Any desired datasets should be placed into a directory called datasets/ from the repository's root directory.

## Data Generation

In order to generate the features from a dataset as described in [1] and [2], run `generate_data.py`, which supports the
following command-line arguments:

- \--dataset The dataset number to perform data generation on. This argument can be 'all' to perform data generation on
  all attributes, 'custom' to perform data generation on a single custom dataset, or the dataset number to perform data
  generation on. The dataset mappings are:
    - 9: UJI Dataset
    - 10: JUIndoor-Loc Dataset
    - 11: UJILoc Dataset
    - 13: IPIN Tutorial Dataset
    - 8: Custom Dataset in the format described in [2]
- \--fingerprints The number of fingerprints to use during data generation. Given that the new features are computed
  between each pair of fingerprints, this argument sets the initial number of fingerprints before pairing. The default
  value for this parameter is set to 100.
- \--upper The upper bound on distance (in meters) for filtering out instances in a middle range (as described in [1]).
  That is, if the upper bound is represented by `x`, then the data generation will filter any instances where the
  distance is between 2.25 meters and `x`.
- \--max The max distance value to be used as a threshold for filtering out instances that have a distance value greater
  than the max distance threshold.
- \--rssi The RSSI signal bound to be used when filtering any RSSI values below the RSSI bound.
- \--processes The number of processes to use for performing data generation in parallel.
- \--custom_dataset An optional argument for providing a custom dataset to be found in the datasets/ directory discussed
  in the Prerequisites section.
- \--custom_file An optional argument for providing the filename for a custom dataset. If the path to the file is nested
  inside another directory from the custom_dataset (e.g., a nested folder for training), this path can be used for
  the argument.

## Converting from CSV to Parquet

In order to convert a generated dataset from CSV to Parquet:

1. Open `convert_to_parquet.py` and uncomment the line below the comment "Uncomment the line below to convert a custom
   dataset"
2. The first argument to this function is a dataset number similar to the Data Generation with the following mappings:
    - 1: The training file of the UJI dataset
    - 2: The testing file of the UJI dataset
    - 3: The testing file of the JUIndoor dataset
    - 4: The training file of the UJILoc dataset
    - 5: The testing file of the UJILoc dataset
    - 6: The training files of the IPIN Tutorial dataset
    - 7: The testing file of the IPIN Tutorial dataset
    - 8: A custom file from a custom dataset
3. [Optional] The second argument is the directory path where the generated dataset CSV file is stored. The current
   example makes use of Python's pathlib library to specify the path to the IPIN dataset's testing directory.
4. [Optional] The third argument is filename of the generated CSV file to be converted **without the file extension**.
5. Finally, run `convert_to_parquet.py`.

## Running the Experiments

In order to run the same experiments discussed in [2], run `run_experiments.py` and the program will prompt for any user
input in the console.

## Server Deployment with Docker

A Dockerfile was included in this repository for deploying and running the experiments on a server since most
experiments can be very time intensive. Note that the Dockerfile sets the working directory to /code-workspace and
assumes that this working directory contains all the source code used in the project as well as the original datasets. The docker file
was then built with the command `docker build -t retsina .` and run
with `docker run -v path_to_mount_output_datasets:output_datasets_in_container -v path_to_mount_output_figs:output_figs_in_container retsina`.
For further resources on the docker commands and mounting volumes, please refer to the official Docker Reference
documentation.

## Citations

- [1] Z. Van Hyfte and A. Zakhor, "Immediate proximity detection using wi-fi-enabled smartphones," in *International
  Conference on Indoor Positioning and Indoor Navigation (IPIN)*, 2021, pp. 1-8.
- [2] Brian T. Nixon, Anna Baskin, Panos K. Chrysanthis, Christos Laoudias, and Constantinos Costa, "RETSINA:
  Reproducibility and Experimentation Testbed for Signal-Strength Indoor Near Analysis," in *International Conference on
  Indoor Positioning and Indoor Navigation (IPIN)*, 2023, pp. ?-?. 
