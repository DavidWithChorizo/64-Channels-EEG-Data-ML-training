# EEG Movement Intention Interpretation

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![GitHub Repo Size](https://img.shields.io/github/repo-size/your-username/EEG-Movement-Intention)
![GitHub Issues](https://img.shields.io/github/issues/your-username/EEG-Movement-Intention)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/your-username/EEG-Movement-Intention)

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Extraction](#data-extraction)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Exploratory Analysis](#exploratory-analysis)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Welcome to the **EEG Movement Intention Interpretation** project! This project aims to develop a machine learning model that accurately interprets the intention of movement from EEG (Electroencephalogram) readings. By analyzing EEG data collected from individuals, the model seeks to predict movement intentions, which has applications in neuroprosthetics, brain-computer interfaces, and cognitive neuroscience research.

## Dataset

### Overview

- **Channels:** 64 EEG channels
- **Sampling Rate:** 150-200 Hz
- **Format:** EDF (European Data Format) with corresponding `.edf.event` files
- **Volume:** 109 separate directories, each containing 14 EDF files and their event files
- **Selected Files:** For each directory, only the 4th, 8th, and 12th EDF files (`SxxxR04.edf`, `SxxxR08.edf`, `SxxxR12.edf`) are used for analysis

### Data Structure
data/
├── raw_data/
│   ├── S001/
│   │   ├── S001R01.edf
│   │   ├── S001R01.edf.event
│   │   ├── …
│   │   ├── S001R14.edf
│   │   └── S001R14.edf.event
│   ├── S002/
│   │   ├── S002R01.edf
│   │   ├── S002R01.edf.event
│   │   ├── …
│   │   ├── S002R14.edf
│   │   └── S002R14.edf.event
│   └── …
└── processed/
├── S001/
│   ├── S001R04_cleaned.csv
│   ├── S001R08_cleaned.csv
│   └── S001R12_cleaned.csv
├── S002/
│   ├── S002R04_cleaned.csv
│   ├── S002R08_cleaned.csv
│   └── S002R12_cleaned.csv
└── …
## Project Structure
EEG-Movement-Intention/
│
├── data/
│   ├── raw_data/            # Original EDF datasets
│   └── processed/           # Cleaned and processed data
│
├── notebooks/               # Jupyter notebooks for analysis
│   ├── exploratory_analysis.ipynb
│   └── model_training.ipynb
│
├── src/                     # Source code
│   ├── data/                # Data preprocessing scripts
│   │   ├── extract_edf_files.py
│   │   └── preprocess_eeg.py
│   ├── models/              # Model training and evaluation scripts
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   └── utils/               # Utility functions
│       └── helper_functions.py
│
├── models/                  # Saved machine learning models
│   ├── eeg_model_v1.h5
│   └── eeg_model_v2.h5
│
├── tests/                   # Unit tests
│   └── test_preprocessing.py
│
├── docs/                    # Documentation
│   └── usage_guide.md
│
├── .gitignore               # Git ignore file
├── README.md                # Project overview
├── requirements.txt         # Python dependencies
├── LICENSE                  # License file
└── setup.py                 # Setup script