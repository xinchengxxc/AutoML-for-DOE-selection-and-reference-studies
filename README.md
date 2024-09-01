# AutoML-for-DOE-selection-and-reference-studies
Repository for the paper "AutoML-based workflow for reliable DOE selection and reference studies for analyzing imprecise experimental data with machine learning"

# Project Name: AutoML and Data Generation Toolkit

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [AutoML Module](#automl-module)
  - [Data Generation Module](#data-generation-module)
  - [Data sets]
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This project provides a set of tools and scripts designed for automated machine learning (AutoML) and data generation tasks. It includes various scripts for running models, preparing data, and automating workflows on high-performance computing (HPC) environments.

Our paper: AutoML-based workflow for reliable DOE selection and reference studies for analyzing imprecise experimental data with machine learning
## Features

- **Automated Machine Learning (AutoML)**: Scripts to automate the process of training and evaluating machine learning models.
- **Data Generation**: Tools for generating and processing large datasets, suitable for simulation or modeling tasks.
- **HPC Integration**: Scripts and notebooks tailored for running in high-performance computing environments.

## Directory Structure

```
autoML/
├── .env
├── .gitignore
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
├── requirements.yaml
├── hpc/
│   ├── auto.bash
│   ├── run_LARs.bash
│   └── out/
├── python/
│   ├── prepare.py
│   ├── run_xxc.py
│   └── Database/
│       └── Modelbase_0.2_0/
│           └── Model81/
└── .devcontainer/
    └── devcontainer.json

data_gen/
├── python/
│   ├── datasets_generation_final_forHPC.py
│   └── Database/
│       └── Modelbase_0.2_0/
│           └── Model81/
└── tools/
    ├── hpctk.py
    ├── hpc_builder.ipynb
    ├── hpc_onboarder.ipynb
    ├── hpc_toolkit.ipynb
    └── assets/
        ├── hpc_onboarder.jpg
        └── logo.jpg
```

## Installation

### Prerequisites

- Docker
- Python 3.8 or higher
- [HPC Toolkit](#)

### Steps

1. Clone the repository:
   \`\`\`bash
   git clone repository.git
   cd repository
   \`\`\`

2. Set up the environment:
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate
   pip install -r autoML/requirements.txt
   \`\`\`

3. (Optional) Set up Docker:
   \`\`\`bash
   docker-compose up
   \`\`\`

## Usage

### AutoML Module

- **Prepare Data**: Use `prepare.py` to preprocess and prepare your data.
- **Run Models**: The purpose of the `run_xxc.py` script is to automate the process of machine learning regression tasks using the `Auto-sklearn` library. It loads experiment configurations, automatically selects and optimizes models, trains and tests on specified datasets, and finally saves the model performance metrics (including R2 score and RMSE) to a designated directory. To use this script, first, prepare an experiment configuration file named `experiment_setup.csv`, and then run the script. The script will automatically execute the machine learning workflow based on the configuration and save the results in the `Experiments/Results_<noise>_<seed>/` directory. This process is designed to facilitate the batch execution of different experiments, particularly in a SLURM cluster environment.

### Data Generation Module

- **Data Generation**: The purpose of the `datasets_generation_final_forHPC.py` script is to automate the generation of datasets for machine learning experiments, particularly in high-performance computing (HPC) environments. The script likely handles the creation and preprocessing of datasets according to specified configurations, preparing them for subsequent machine-learning tasks. To use this script, ensure that the necessary configurations are set, and then execute the script. It will generate the datasets as defined and save them to the appropriate directories, ready for use in various machine learning experiments, especially when batch processing on HPC clusters is required.

### Data sets

- This project contains multiple database folders, each containing sets of model base files (`Modelbase`). These databases and model files are used for experiments and data analysis, with different noise levels and multiple sets of repeated experimental data. Only a complete example dataset has been uploaded to this repository due to the number of files and size considerations. Any dataset presented in this work can be generated using the provided code.

## Configuration

Several configuration files are included:

- **`.env`**: Environment variables configuration.
- **`__hpc_creds.yaml`**: HPC credentials configuration.
- **`.vscode/`**: VSCode-specific settings and launch configurations.

Ensure that you modify these files according to your environment and requirements.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

## Contact
Xukuan.Xu@th-ab.de
xukuan xu(徐旭宽)
