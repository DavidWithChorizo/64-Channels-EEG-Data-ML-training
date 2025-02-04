#!/usr/bin/env python
"""
A complete EEG motor-imagery ML software that:
- Loads EDF files from a raw data directory (e.g., S001R04.edf).
- Processes and epochs the data using MNE.
- Applies multiple band-pass filters to cover a wider frequency range.
- Uses a Filter Bank CSP (FBCSP) approach (split into FilterBankTransformer + CSP).
- Builds an ML pipeline with a RandomForestClassifier.
- Performs hyperparameter tuning using GridSearchCV.
- Evaluates the overall accuracy on a held-out test set.

Usage:
    python script_name.py
"""

import os
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from mne import Epochs
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV  # <-- NEW IMPORT

###############################################################################
# LOGGING CONFIGURATION
###############################################################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############################################################################
# (Optional) CHANNEL SELECTOR TRANSFORMER
###############################################################################
class ChannelSelector(BaseEstimator, TransformerMixin):
    """
    Selects a subset of channels from EEG data.
    
    Parameters
    ----------
    channels : list of int
        List of channel indices to keep.
    """
    def __init__(self, channels):
        self.channels = channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Assuming X shape: (n_epochs, n_channels, n_times)
        return X[:, self.channels, :]

###############################################################################
# FILTER BANK TRANSFORMER
###############################################################################
class FilterBankTransformer(BaseEstimator, TransformerMixin):
    """
    Applies multiple band-pass filters to EEG data and returns a list of filtered signals.
    Each item in the list corresponds to one frequency band.

    Parameters
    ----------
    freq_bands : list of tuple
        Each tuple is (l_freq, h_freq), the lower and upper frequency for a band-pass filter.
    sfreq : float
        Sampling frequency of the EEG data.
    filter_method : str
        Method to use for filtering ('iir' or 'fir').
    """
    def __init__(self, freq_bands, sfreq=160, filter_method='iir'):
        self.freq_bands = freq_bands
        self.sfreq = sfreq
        self.filter_method = filter_method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.ndim != 3:
            raise ValueError(f"X should be 3D (n_trials, n_channels, n_times). Got shape: {X.shape}")
        X_filtered_list = []
        for (l_freq, h_freq) in self.freq_bands:
            logging.info(f"Filtering data for band: {l_freq}-{h_freq} Hz")
            X_filtered = np.array([
                mne.filter.filter_data(trial,
                                       sfreq=self.sfreq,
                                       l_freq=l_freq,
                                       h_freq=h_freq,
                                       method=self.filter_method,
                                       verbose=False)
                for trial in X
            ])
            X_filtered_list.append(X_filtered)
        return X_filtered_list

###############################################################################
# FBCSP TRANSFORMER (Adapted)
###############################################################################
class FBCSP(BaseEstimator, TransformerMixin):
    """
    Simplified FBCSP that expects data already filtered per frequency band.
    Applies an MNE CSP transformer to each band, extracting log-variance features.
    
    Parameters
    ----------
    n_csp : int, optional
        Number of CSP components per band (default is 4).
    csp_reg : None or str or float, optional
        Regularization parameter for CSP.
    log : bool, optional
        If True, the CSP transformer will compute the log variance.
    """
    def __init__(self, n_csp=4, csp_reg=None, log=True):
        self.n_csp = n_csp
        self.csp_reg = csp_reg
        self.log = log
        self.csp_list_ = []

    def fit(self, X_list, y):
        self.csp_list_ = []
        for idx, X_band in enumerate(X_list):
            logging.info(f"Fitting CSP for band index {idx}")
            csp = CSP(n_components=self.n_csp, reg=self.csp_reg, log=self.log, norm_trace=False)
            csp.fit(X_band, y)
            self.csp_list_.append(csp)
        return self

    def transform(self, X_list):
        if not self.csp_list_:
            raise RuntimeError("FBCSP has not been fitted yet. Call fit before transform.")
        features = []
        for idx, X_band in enumerate(X_list):
            logging.info(f"Transforming data for band index {idx}")
            csp = self.csp_list_[idx]
            X_csp = csp.transform(X_band)
            features.append(X_csp)
        X_features = np.concatenate(features, axis=1)
        return X_features

###############################################################################
# DATA LOADING AND PREPROCESSING FUNCTIONS
###############################################################################
def load_file_info(data_dir, desired_runs=('04', '08', '12')):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"The specified data directory does not exist: {data_dir}")
    edf_pattern = re.compile(r'^S\d{3}R0?(' + '|'.join(desired_runs) + r')\.edf$', re.IGNORECASE)
    file_info = []
    for subject_dir in data_dir.iterdir():
        if subject_dir.is_dir() and re.match(r'^S\d{3}$', subject_dir.name, re.IGNORECASE):
            for file in subject_dir.iterdir():
                if file.is_file() and edf_pattern.match(file.name):
                    file_info.append({
                        'subject': subject_dir.name,
                        'edf_file': file.name,
                        'edf_path': str(file.resolve())
                    })
    df_files = pd.DataFrame(file_info)
    logging.info(f"Total matched .edf files: {df_files.shape[0]}")
    return df_files

def process_edf_file(row, event_dict, tmin=-0.2, tmax=3.8, baseline=(None, 0)):
    edf_path = row['edf_path']
    subject = row['subject']
    edf_file = row['edf_file']
    logging.info(f"Processing file: {edf_file} for subject: {subject}")
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        events, _ = mne.events_from_annotations(raw, event_id=event_dict)
        if events.size == 0:
            logging.warning(f"No events found in {edf_file}. Skipping.")
            return None, None
        epochs = Epochs(raw, events, event_id=event_dict,
                        tmin=tmin, tmax=tmax,
                        baseline=baseline,
                        preload=True, verbose=False)
        motor_epochs = epochs[['T1', 'T2']]
        if len(motor_epochs) == 0:
            logging.warning(f"No motor-related epochs found in {edf_file}. Skipping.")
            return None, None
        labels = motor_epochs.events[:, -1] - event_dict['T1']
        X = motor_epochs.get_data()
        y = motor_epochs.events[:, -1] - event_dict['T1']
        expected_n_times = 641
        if X.shape[-1] != expected_n_times:
            logging.warning(
                f"Skipping file {row['edf_file']} due to mismatch: found {X.shape[-1]} time samples instead of {expected_n_times}."
            )
            return None, None
        return X, y
    except Exception as e:
        logging.error(f"Failed to process {edf_file}. Error: {e}")
        return None, None

def load_all_data(df_files, event_dict, tmin=-0.2, tmax=3.8, baseline=(None, 0)):
    all_epochs = []
    all_labels = []
    for idx, row in df_files.iterrows():
        X, y = process_edf_file(row, event_dict, tmin=tmin, tmax=tmax, baseline=baseline)
        if X is not None and y is not None:
            all_epochs.append(X)
            all_labels.append(y)
    if not all_epochs:
        raise RuntimeError("No valid epochs were loaded. Check your data and event labels.")
    X_all = np.concatenate(all_epochs, axis=0)
    y_all = np.concatenate(all_labels, axis=0)
    logging.info(f"Final X_all shape: {X_all.shape}")
    logging.info(f"Final y_all shape: {y_all.shape}")
    return X_all, y_all

###############################################################################
# ML PIPELINE SETUP
###############################################################################
def build_pipeline(freq_bands, n_csp, sfreq, classifier_params=None, selected_channels=None):
    """
    Build a machine learning pipeline with optional channel selection, filtering, CSP, and classification.
    
    Parameters
    ----------
    freq_bands : list of tuple
        Frequency bands for filtering.
    n_csp : int
        Number of CSP components per band.
    sfreq : float
        Sampling frequency.
    classifier_params : dict, optional
        Parameters for the RandomForestClassifier.
    selected_channels : list of int, optional
        Indices of channels to keep. If provided, ChannelSelector will be used.
        
    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        The constructed ML pipeline.
    """
    if classifier_params is None:
        classifier_params = {}
    
    steps = []
    if selected_channels is not None:
        steps.append(('channel_selector', ChannelSelector(selected_channels)))
    
    steps.extend([
        ('filter_bank', FilterBankTransformer(freq_bands=freq_bands, sfreq=sfreq, filter_method='iir')),
        ('fbcsp', FBCSP(n_csp=n_csp, csp_reg=None, log=True)),
        ('clf', RandomForestClassifier(**classifier_params))
    ])
    
    pipeline = Pipeline(steps)
    return pipeline

###############################################################################
# EVALUATION FUNCTION
###############################################################################
def evaluate_pipeline(pipeline, X, y):
    """
    Evaluate the ML pipeline by predicting on the data and logging the accuracy.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The ML pipeline (should already be fitted).
    X : np.ndarray
        Input data, shape: (n_epochs, n_channels, n_times).
    y : np.ndarray
        True labels for each trial.
        
    Returns
    -------
    accuracy : float
        Overall accuracy (from 0 to 1).
    """
    y_pred = pipeline.predict(X)
    accuracy = accuracy_score(y, y_pred)
    logging.info(f"Accuracy on the test set: {accuracy * 100:.2f}%")
    return accuracy

###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    """
    Main function to execute the EEG motor-imagery ML pipeline.
    
    Steps:
    1) Define paths, event dict, and parameters.
    2) Load file information from the raw data directory.
    3) Process and load all EEG data and corresponding labels.
    4) Split the data into training and test sets.
    5) Build the machine learning pipeline.
    6) Perform hyperparameter tuning with GridSearchCV.
    7) Evaluate the tuned pipeline on the test dataset.
    """
    # 1. Define paths, event dict, and parameters
    data_directory = './data/raw data'  # Adjust to your actual data path
    desired_runs = ('04', '08', '12')
    event_dict = {'T0': 1, 'T1': 2, 'T2': 3}

    # Epoch parameters
    tmin, tmax, baseline = -0.2, 3.8, (None, 0)

    # Sampling frequency
    sfreq = 160

    # Define a wider set of frequency bands for the FilterBankTransformer.
    # For example, we cover from 4 Hz to 40 Hz in 4 Hz steps.
    freq_bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]

    # Number of CSP components
    n_csp = 4

    # Classifier parameters for RandomForestClassifier (default settings)
    classifier_params = {
        'random_state': 42
    }

    # (Optional) Selected channel indices if you want to reduce channels. None = use all channels.
    selected_channels = None  # Change to a list (e.g., [0,1,2,3,4,5,6,7]) if you want to select a subset.

    # 2. Load file information
    df_files = load_file_info(data_directory, desired_runs=desired_runs)
    if df_files.empty:
        raise RuntimeError("No matching EDF files found. Check your data or regex pattern.")

    # 3. Process and load all EEG data
    X_all, y_all = load_all_data(df_files, event_dict, tmin=tmin, tmax=tmax, baseline=baseline)

    # 4. Split the data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # 5. Build the machine learning pipeline
    pipeline = build_pipeline(freq_bands, n_csp, sfreq, classifier_params, selected_channels)

    # 6. Hyperparameter tuning with GridSearchCV  <-- NEW CODE
    # Define a parameter grid. Here we vary the number of trees and maximum depth.
    param_grid = {
        'clf__n_estimators': [100, 150],
        'clf__max_depth': [None, 5, 10],
        # You could also tune parameters of FBCSP or other steps if desired.
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    logging.info("Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_ * 100:.2f}%")

    # 7. Evaluate the tuned pipeline on the test dataset.
    best_pipeline = grid_search.best_estimator_
    accuracy = evaluate_pipeline(best_pipeline, X_test, y_test)
    logging.info(f"Final test accuracy with tuned parameters: {accuracy * 100:.2f}%")

# Entry point
if __name__ == '__main__':
    main()