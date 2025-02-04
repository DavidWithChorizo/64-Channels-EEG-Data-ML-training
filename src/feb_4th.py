#!/usr/bin/env python
"""
A streamlined EEG motor-imagery ML pipeline that:
- Loads and preprocesses EEG data.
- Uses multiple band-pass filters (with a limited set of frequency bands).
- Applies a Filter Bank CSP (FBCSP) transformation.
- Builds an ML pipeline with a RandomForestClassifier.
- Performs manual hyperparameter tuning using a simple hold-out validation set
  (instead of GridSearchCV) to reduce memory usage.
- Evaluates the final model on a held-out test set.

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
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
        # Assumes X shape: (n_epochs, n_channels, n_times)
        return X[:, self.channels, :]

###############################################################################
# FILTER BANK TRANSFORMER
###############################################################################
class FilterBankTransformer(BaseEstimator, TransformerMixin):
    """
    Applies multiple band-pass filters to EEG data and returns a list of filtered signals.
    Each element in the list corresponds to one frequency band.
    
    Parameters
    ----------
    freq_bands : list of tuple
        Each tuple is (l_freq, h_freq) for the band-pass filter.
    sfreq : float
        Sampling frequency of the EEG data.
    filter_method : str
        Filtering method ('iir' or 'fir').
    """
    def __init__(self, freq_bands, sfreq=160, filter_method='iir'):
        self.freq_bands = freq_bands
        self.sfreq = sfreq
        self.filter_method = filter_method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.ndim != 3:
            raise ValueError(f"Expected X to be 3D (n_trials, n_channels, n_times), got shape: {X.shape}")
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
    Simplified FBCSP that applies CSP to each frequency band and extracts log-variance features.
    
    Parameters
    ----------
    n_csp : int, optional
        Number of CSP components per band (default is 4).
    csp_reg : None, str, or float, optional
        Regularization parameter for CSP.
    log : bool, optional
        If True, compute the log variance.
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
        raise FileNotFoundError(f"The data directory does not exist: {data_dir}")
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
        X = motor_epochs.get_data()
        y = motor_epochs.events[:, -1] - event_dict['T1']
        expected_n_times = 641
        if X.shape[-1] != expected_n_times:
            logging.warning(f"File {edf_file} skipped due to mismatch in time samples.")
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
        raise RuntimeError("No valid epochs were loaded.")
    X_all = np.concatenate(all_epochs, axis=0)
    y_all = np.concatenate(all_labels, axis=0)
    logging.info(f"Final X_all shape: {X_all.shape}")
    logging.info(f"Final y_all shape: {y_all.shape}")
    return X_all, y_all

###############################################################################
# PIPELINE SETUP
###############################################################################
def build_pipeline(freq_bands, n_csp, sfreq, classifier_params=None, selected_channels=None):
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
    return Pipeline(steps)

def evaluate_pipeline(pipeline, X, y):
    y_pred = pipeline.predict(X)
    accuracy = accuracy_score(y, y_pred)
    logging.info(f"Accuracy on the test set: {accuracy * 100:.2f}%")
    return accuracy

###############################################################################
# MAIN FUNCTION WITH MANUAL HYPERPARAMETER TUNING
###############################################################################
def main():
    # 1. Define paths, event dictionary, and parameters
    data_directory = './data/raw data'  # Adjust to your actual data path
    desired_runs = ('04', '08', '12')
    event_dict = {'T0': 1, 'T1': 2, 'T2': 3}
    
    tmin, tmax, baseline = -0.2, 3.8, (None, 0)
    sfreq = 160

    # Use a limited set of frequency bands for efficiency
    freq_bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36)]
    
    n_csp = 4
    classifier_params = {'random_state': 42}
    selected_channels = None  # Use all channels

    # 2. Load file information and data
    df_files = load_file_info(data_directory, desired_runs=desired_runs)
    if df_files.empty:
        raise RuntimeError("No matching EDF files found.")
    X_all, y_all = load_all_data(df_files, event_dict, tmin=tmin, tmax=tmax, baseline=baseline)
    
    # 3. Split the data into training and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    logging.info(f"Training+Validation set shape: {X_train_full.shape}, Test set shape: {X_test.shape}")
    
    # 4. Further split the training data into training and validation sets (for tuning)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    logging.info(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
    
    # 5. Build the base pipeline (without tuning)
    base_pipeline = build_pipeline(freq_bands, n_csp, sfreq, classifier_params, selected_channels)
    
    # 6. Manual hyperparameter tuning using a small grid and the validation set.
    best_score = 0.0
    best_params = None
    best_pipeline = None

    # Define a small grid of parameters to try.
    for n_estimators in [100, 200]:
        for max_depth in [None, 10]:
            logging.info(f"Testing parameters: n_estimators={n_estimators}, max_depth={max_depth}")
            # Clone the base pipeline to avoid reusing the same model instance
            candidate_pipeline = clone(base_pipeline)
            candidate_pipeline.set_params(clf__n_estimators=n_estimators, clf__max_depth=max_depth)
            candidate_pipeline.fit(X_train, y_train)
            score = candidate_pipeline.score(X_val, y_val)
            logging.info(f"Validation score: {score * 100:.2f}%")
            if score > best_score:
                best_score = score
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
                best_pipeline = candidate_pipeline

    logging.info(f"Best parameters found: {best_params} with validation score: {best_score * 100:.2f}%")
    
    # 7. Retrain best model on the entire training+validation set
    final_pipeline = clone(base_pipeline)
    final_pipeline.set_params(clf__n_estimators=best_params['n_estimators'],
                                clf__max_depth=best_params['max_depth'])
    final_pipeline.fit(X_train_full, y_train_full)
    
    # 8. Evaluate on the held-out test set
    test_accuracy = evaluate_pipeline(final_pipeline, X_test, y_test)
    logging.info(f"Final test accuracy: {test_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()