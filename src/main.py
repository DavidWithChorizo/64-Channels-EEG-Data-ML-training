#!/usr/bin/env python
"""
A complete EEG motor-imagery ML software that:
- Loads EDF files from a raw data directory (e.g., S001R04.edf).
- Processes and epochs the data using MNE.
- Applies separate filter bank transformations for multiple frequency bands.
- Uses a Filter Bank CSP (FBCSP) approach (now split into FilterBankTransformer + CSP).
- Builds an ML pipeline with an MLPClassifier.
- Evaluates the overall accuracy.

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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

###############################################################################
# LOGGING CONFIGURATION
###############################################################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        """
        No fitting needed for a simple filter step.
        """
        return self

    def transform(self, X):
        """
        Band-pass filter the data for each specified frequency band.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        X_filtered_list : list of np.ndarray
            Each element in the list corresponds to one frequency band.
            Each element has shape (n_trials, n_channels, n_times).
        """
        if X.ndim != 3:
            raise ValueError(f"X should be 3D (n_trials, n_channels, n_times). Got shape: {X.shape}")

        X_filtered_list = []
        for (l_freq, h_freq) in self.freq_bands:
            logging.info(f"Filtering data for band: {l_freq}-{h_freq} Hz")
            # Filter once per band
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

    It applies an MNE CSP transformer to each band, extracting log-variance features
    (if CSP is initialized with log=True).

    Parameters
    ----------
    n_csp : int, optional
        Number of CSP components per band (default is 4).
    csp_reg : None or str or float, optional
        Regularization parameter for CSP. Examples: None, 'ledoit_wolf', 'oas', or a float value.
    log : bool, optional
        If True, the CSP transformer will automatically compute the log variance.
    """
    def __init__(self, n_csp=4, csp_reg=None, log=True):
        self.n_csp = n_csp
        self.csp_reg = csp_reg
        self.log = log
        self.csp_list_ = []

    def fit(self, X_list, y):
        """
        Fit CSP models for each pre-filtered frequency band.

        Parameters
        ----------
        X_list : list of np.ndarray
            Each element is (n_trials, n_channels, n_times) for a specific frequency band.
        y : np.ndarray, shape (n_trials,)
            Class labels.

        Returns
        -------
        self : object
            The fitted FBCSP transformer.
        """
        self.csp_list_ = []
        for idx, X_band in enumerate(X_list):
            logging.info(f"Fitting CSP for band index {idx}")
            # Initialize and fit CSP for the current band
            csp = CSP(n_components=self.n_csp, reg=self.csp_reg, log=self.log, norm_trace=False)
            csp.fit(X_band, y)
            self.csp_list_.append(csp)

        return self

    def transform(self, X_list):
        """
        Transform EEG data using the fitted CSP models, extracting features.

        Parameters
        ----------
        X_list : list of np.ndarray
            Each element is (n_trials, n_channels, n_times) for a specific frequency band.

        Returns
        -------
        X_features : np.ndarray, shape (n_trials, n_bands * n_csp)
            Extracted features concatenated across bands.
        """
        if not self.csp_list_:
            raise RuntimeError("FBCSP has not been fitted yet. Call fit before transform.")

        features = []
        for idx, X_band in enumerate(X_list):
            logging.info(f"Transforming data for band index {idx}")
            csp = self.csp_list_[idx]

            # Apply CSP transformation (log variance if log=True)
            X_csp = csp.transform(X_band)  # shape: (n_trials, n_components)
            features.append(X_csp)

        # Concatenate features across frequency bands
        X_features = np.concatenate(features, axis=1)  # (n_trials, n_bands*n_csp)
        return X_features


###############################################################################
# DATA LOADING AND PREPROCESSING FUNCTIONS
###############################################################################
def load_file_info(data_dir, desired_runs=('04', '08', '12')):
    """
    Scan the raw data directory and collect file information for EDF files.

    Parameters
    ----------
    data_dir : str or Path
        Path to the directory containing subject subdirectories.
    desired_runs : tuple of str, optional
        List of run identifiers to include (default is ('04', '08', '12')).

    Returns
    -------
    df_files : pandas.DataFrame
        DataFrame containing file metadata for each valid EDF file.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"The specified data directory does not exist: {data_dir}")

    # Regex to match e.g. S001R04.edf, S002R08.edf, S003R12.edf
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
    """
    Process a single EDF file to load and epoch EEG data.

    Parameters
    ----------
    row : pandas.Series
        A row from the file info DataFrame containing subject name, file paths, etc.
    event_dict : dict
        Mapping from annotation keys to numeric event codes (e.g., {'T0': 1, 'T1': 2, 'T2': 3}).
    tmin : float, optional
        Start time (in seconds) before the event (default is -0.2).
    tmax : float, optional
        End time (in seconds) after the event (default is 3.8).
    baseline : tuple or None, optional
        Baseline correction interval (default is (None, 0)).

    Returns
    -------
    X : np.ndarray or None
        Epoched EEG data (n_epochs, n_channels, n_times), or None if processing fails.
    y : np.ndarray or None
        Corresponding labels for each epoch, or None if processing fails.
    """
    edf_path = row['edf_path']
    subject = row['subject']
    edf_file = row['edf_file']

    logging.info(f"Processing file: {edf_file} for subject: {subject}")
    try:
        # Read EDF and create raw object
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        # Convert annotations to events using the provided event_dict
        events, _ = mne.events_from_annotations(raw, event_id=event_dict)

        if events.size == 0:
            logging.warning(f"No events found in {edf_file}. Skipping.")
            return None, None

        # Create epochs around the events
        epochs = Epochs(raw, events, event_id=event_dict,
                        tmin=tmin, tmax=tmax,
                        baseline=baseline,
                        preload=True, verbose=False)

        # Select only motor imagery epochs (here T1 and T2)
        # T1: Imagining Left Hand
        # T2: Imagining Right Hand
        motor_epochs = epochs[['T1', 'T2']]
        if len(motor_epochs) == 0:
            logging.warning(f"No motor-related epochs found in {edf_file}. Skipping.")
            return None, None

        # Extract labels: 0 for 'T1', 1 for 'T2' (assuming event_dict={'T0':1,'T1':2,'T2':3})
        labels = motor_epochs.events[:, -1] - event_dict['T1']  # 2 -> class 0, 3 -> class 1

        # Retrieve epoched data
        X = motor_epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        y = labels.astype(int)
        logging.info(f"Loaded {X.shape[0]} epochs from {edf_file}.")
        return X, y

    except Exception as e:
        logging.error(f"Failed to process {edf_file}. Error: {e}")
        return None, None


def load_all_data(df_files, event_dict, tmin=-0.2, tmax=3.8, baseline=(None, 0)):
    """
    Process all EDF files from the file info DataFrame and concatenate the results.

    Parameters
    ----------
    df_files : pandas.DataFrame
        DataFrame containing metadata and paths for each EDF file.
    event_dict : dict
        Mapping from annotation keys to numeric event codes.
    tmin : float
        Start time for epoching (seconds).
    tmax : float
        End time for epoching (seconds).
    baseline : tuple or None
        Baseline correction interval.

    Returns
    -------
    X_all : np.ndarray
        Concatenated EEG data (n_total_epochs, n_channels, n_times).
    y_all : np.ndarray
        Concatenated labels (n_total_epochs,).

    Raises
    ------
    RuntimeError
        If no valid epochs were loaded from any file.
    """
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
def build_pipeline(freq_bands, n_csp, sfreq, classifier_params=None):
    """
    Build a machine learning pipeline with separate filtering and CSP steps.

    1) FilterBankTransformer: filters data into multiple bands
    2) FBCSP: applies CSP for each band and extracts features
    3) MLPClassifier: classifies the extracted features

    Parameters
    ----------
    freq_bands : list of tuple
        List of frequency bands for filtering (e.g., [(8,12), (12,16), (16,20)]).
    n_csp : int
        Number of CSP components per band.
    sfreq : float
        Sampling frequency of the EEG data.
    classifier_params : dict, optional
        Parameters for the MLPClassifier.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        The constructed ML pipeline.
    """
    if classifier_params is None:
        classifier_params = {}  # default parameters if none provided

    pipeline = Pipeline([
        ('filter_bank', FilterBankTransformer(freq_bands=freq_bands,
                                             sfreq=sfreq,
                                             filter_method='iir')),
        ('fbcsp', FBCSP(n_csp=n_csp, csp_reg=None, log=True)),
        ('clf', MLPClassifier(**classifier_params))
    ])
    return pipeline


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
    logging.info(f"Accuracy on the entire dataset: {accuracy * 100:.2f}%")
    return accuracy


###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    """
    Main function to execute the EEG motor-imagery ML pipeline.

    1) Define paths, event dict, and parameters.
    2) Load file information from the raw data directory.
    3) Process and load all EEG data and corresponding labels.
    4) Build the machine learning pipeline.
    5) Fit the pipeline on the entire dataset.
    6) Evaluate the pipeline on the same dataset.
    """
    # 1. Define paths, event dict, and parameters
    data_directory = '../data/raw data'  # Adjust to your actual data path
    desired_runs = ('04', '08', '12')    # For example: 04, 08, 12
    event_dict = {'T0': 1, 'T1': 2, 'T2': 3}  # Adjust if your annotations differ

    # Epoch parameters
    tmin, tmax, baseline = -0.2, 3.8, (None, 0)

    # Sampling frequency (adjust as needed)
    sfreq = 160

    # Frequency bands for FilterBankTransformer
    freq_bands = [(8, 12), (12, 16), (16, 20)]

    # Number of CSP components
    n_csp = 4

    # Classifier parameters (for MLPClassifier)
    classifier_params = {
        'hidden_layer_sizes': (100,),
        'max_iter': 300,
        'random_state': 42
    }

    # 2. Load file information
    df_files = load_file_info(data_directory, desired_runs=desired_runs)
    if df_files.empty:
        raise RuntimeError("No matching EDF files found. Check your data or regex pattern.")

    # 3. Process and load all EEG data
    X_all, y_all = load_all_data(df_files, event_dict, tmin=tmin, tmax=tmax, baseline=baseline)

    # 4. Build the machine learning pipeline
    pipeline = build_pipeline(freq_bands, n_csp, sfreq, classifier_params)

    # 5. Fit the pipeline on the entire dataset
    logging.info("Fitting the pipeline on the entire dataset...")
    pipeline.fit(X_all, y_all)

    # 6. Evaluate the pipeline
    accuracy = evaluate_pipeline(pipeline, X_all, y_all)
    logging.info(f"Final accuracy: {accuracy * 100:.2f}%")

# Entry point
if __name__ == '__main__':
    main()