#!/usr/bin/env python
"""
A complete EEG motor-imagery ML software:
- Loads EDF files (and matching event files) from a raw data directory.
- Processes and epochs the data.
- Extracts features using a custom Filter Bank Common Spatial Pattern (FBCSP) transformer.
- Builds an ML pipeline with an MLPClassifier.
- Evaluates the overall accuracy.
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
from mne.filter import filter_data
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Configure logging for debugging and info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


###############################################################################
# CUSTOM TRANSFORMER: Filter Bank Common Spatial Pattern (FBCSP)
###############################################################################
class FBCSP(BaseEstimator, TransformerMixin):
    """
    Filter Bank Common Spatial Pattern (FBCSP) transformer.

    Applies a series of band-pass filters to EEG data, fits a Common Spatial Pattern (CSP)
    for each frequency band, and then transforms the data to a feature space via log-variance
    of the CSP components.

    Parameters
    ----------
    freq_bands : list of tuple
        List of frequency bands (l_freq, h_freq) to filter the data.
    n_csp : int, optional
        Number of CSP components per band (default is 4).
    sfreq : float, optional
        Sampling frequency of the EEG data (default is 160 Hz).
    filter_method : str, optional
        Filtering method to use ('iir' or 'fir', default is 'iir').
    csp_reg : None or str or float, optional
        Regularization parameter for CSP. Examples: None, 'ledoit_wolf', 'oas', or a float value.
    """
    def __init__(self, freq_bands, n_csp=4, sfreq=160, filter_method='iir', csp_reg=None):
        self.freq_bands = freq_bands
        self.n_csp = n_csp
        self.sfreq = sfreq
        self.filter_method = filter_method
        self.csp_reg = csp_reg
        self.csp_list_ = []

    def fit(self, X, y):
        """
        Fit CSP models for each specified frequency band.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)
            EEG data.
        y : np.ndarray, shape (n_trials,)
            Class labels.

        Returns
        -------
        self : object
            The fitted FBCSP transformer.
        """
        if X.ndim != 3:
            raise ValueError(f"X should be 3D (n_trials, n_channels, n_times). Got shape: {X.shape}")
        if X.shape[0] != len(y):
            raise ValueError("Number of trials in X does not match length of y.")

        self.csp_list_ = []
        for band in self.freq_bands:
            l_freq, h_freq = band
            logging.info(f"Fitting CSP for band: {l_freq}-{h_freq} Hz")

            # Band-pass filter the data for the current frequency band
            X_filtered = np.array([
                filter_data(trial, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq,
                            method=self.filter_method, verbose=False)
                for trial in X
            ])

            # Initialize and fit CSP for the current band
            csp = CSP(n_components=self.n_csp, reg=self.csp_reg, log=True, norm_trace=False)
            csp.fit(X_filtered, y)
            self.csp_list_.append(csp)
        return self

    def transform(self, X):
        """
        Transform EEG data using the fitted CSP models, extracting log-variance features.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)
            EEG data to transform.

        Returns
        -------
        X_features : np.ndarray, shape (n_trials, n_bands * n_csp)
            Extracted features.
        """
        if X.ndim != 3:
            raise ValueError(f"X should be 3D (n_trials, n_channels, n_times). Got shape: {X.shape}")
        if not self.csp_list_:
            raise RuntimeError("FBCSP has not been fitted yet. Call fit before transform.")

        features = []
        for idx, (l_freq, h_freq) in enumerate(self.freq_bands):
            logging.info(f"Transforming data for band: {l_freq}-{h_freq} Hz")
            csp = self.csp_list_[idx]

            X_filtered = np.array([
                filter_data(trial, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq,
                            method=self.filter_method, verbose=False)
                for trial in X
            ])

            # Apply CSP transformation
            X_csp = csp.transform(X_filtered)
            # Extract log-variance features
            X_logvar = np.log(np.var(X_csp, axis=1)).reshape(-1, 1)
            features.append(X_logvar)
        X_features = np.column_stack(features)
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

    # Regular expression to match files (e.g., S001R04.edf, S002R08.edf, S001R012.edf)
    edf_pattern = re.compile(r'^S\d{3}R0?(' + '|'.join(desired_runs) + r')\.edf$', re.IGNORECASE)
    file_info = []
    processed_subjects = 0

    for subject_dir in data_dir.iterdir():
        if subject_dir.is_dir() and re.match(r'^S\d{3}$', subject_dir.name, re.IGNORECASE):
            processed_subjects += 1
            for file in subject_dir.iterdir():
                if file.is_file() and edf_pattern.match(file.name):
                    base_name = file.stem  # e.g., S001R04
                    event_file = f"{base_name}.edf.event"
                    event_path = subject_dir / event_file
                    event_exists = event_path.exists()
                    file_info.append({
                        'subject': subject_dir.name,
                        'edf_file': file.name,
                        'event_file': event_file,
                        'event_exists': event_exists,
                        'edf_path': str(file.resolve()),
                        'event_path': str(event_path.resolve()) if event_exists else None
                    })

    df_files = pd.DataFrame(file_info)
    logging.info(f"Total subject directories processed: {processed_subjects}")
    logging.info(f"Total matched .edf files: {df_files.shape[0]}")
    logging.info(f"Number of corresponding event files found: {df_files['event_exists'].sum()}")
    missing_events = df_files.shape[0] - df_files['event_exists'].sum()
    logging.info(f"Number of missing event files: {missing_events}")
    return df_files


def process_edf_file(row, event_dict, tmin=-0.2, tmax=3.8, baseline=(None, 0)):
    """
    Process a single EDF file to load and epoch EEG data.

    Parameters
    ----------
    row : pandas.Series
        A row from the file info DataFrame containing paths and metadata.
    event_dict : dict
        Mapping from annotation keys to numeric event codes.
    tmin : float, optional
        Start time (in seconds) before the event (default is -0.2).
    tmax : float, optional
        End time (in seconds) after the event (default is 3.8).
    baseline : tuple or None, optional
        Baseline correction interval (default is (None, 0)).

    Returns
    -------
    X : np.ndarray or None
        The epoched EEG data of shape (n_epochs, n_channels, n_times) or None if processing fails.
    y : np.ndarray or None
        The corresponding labels for each epoch, or None if processing fails.
    """
    edf_path = row['edf_path']
    subject = row['subject']
    edf_file = row['edf_file']
    event_exists = row['event_exists']

    logging.info(f"Processing file: {edf_file} for subject: {subject}")
    if not event_exists:
        logging.warning(f"Skipping {edf_file} as it lacks an event file.")
        return None, None

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        # Convert annotations to events using the provided event_dict
        events, _ = mne.events_from_annotations(raw, event_id=event_dict)
        if events.size == 0:
            logging.warning(f"No events found in {edf_file}. Skipping.")
            return None, None

        # Create epochs around the events
        epochs = Epochs(raw, events, event_id=event_dict, tmin=tmin, tmax=tmax,
                        baseline=baseline, preload=True, verbose=False)
        # Select only motor imagery epochs (here T1 and T2)
        motor_epochs = epochs[['T1', 'T2']]
        if len(motor_epochs) == 0:
            logging.warning(f"No motor-related epochs found in {edf_file}. Skipping.")
            return None, None

        # Extract labels: 0 for 'T1' (Imagining Left Hand), 1 for 'T2' (Imagining Right Hand)
        labels = motor_epochs.events[:, -1] - event_dict['T1']
        X = motor_epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        y = labels
        logging.info(f"Loaded {X.shape[0]} epochs from {edf_file}.")
        return X, y
    except Exception as e:
        logging.error(f"Failed to process {edf_file}. Error: {e}")
        return None, None


def load_all_data(df_files, event_dict):
    """
    Process all EDF files from the file info DataFrame and concatenate the results.

    Parameters
    ----------
    df_files : pandas.DataFrame
        DataFrame containing metadata and paths for each EDF file.
    event_dict : dict
        Mapping from annotation keys to numeric event codes.

    Returns
    -------
    X_all : np.ndarray
        Concatenated EEG data from all files, shape (total_epochs, n_channels, n_times).
    y_all : np.ndarray
        Concatenated labels from all files, shape (total_epochs,).

    Raises
    ------
    RuntimeError
        If no valid epochs were loaded from any file.
    """
    all_epochs = []
    all_labels = []
    for idx, row in df_files.iterrows():
        X, y = process_edf_file(row, event_dict)
        if X is not None and y is not None:
            all_epochs.append(X)
            all_labels.append(y)
    if not all_epochs:
        raise RuntimeError("No valid epochs were loaded. Check your data and event files.")
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
    Build a machine learning pipeline with the FBCSP transformer and an MLPClassifier.

    Parameters
    ----------
    freq_bands : list of tuple
        List of frequency bands for FBCSP.
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
        classifier_params = {}  # default parameters for MLPClassifier
    clf = MLPClassifier(**classifier_params)
    pipeline = Pipeline([
        ('fbcsp', FBCSP(freq_bands=freq_bands, n_csp=n_csp, sfreq=sfreq)),
        ('clf', clf)
    ])
    return pipeline


def evaluate_pipeline(pipeline, X, y):
    """
    Evaluate the ML pipeline by predicting on the data and logging the accuracy.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The ML pipeline (should be already fitted).
    X : np.ndarray
        The input data.
    y : np.ndarray
        The true labels.

    Returns
    -------
    accuracy : float
        The overall accuracy of the predictions.
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

    Steps:
    1. Define paths and parameters.
    2. Load file information from the raw data directory.
    3. Process and load all EEG data and corresponding labels.
    4. Build the machine learning pipeline.
    5. Fit the pipeline on the entire dataset.
    6. Evaluate the pipeline.
    """
    # 1. Define paths and parameters
    data_directory = '../data/raw data'  # Adjust to your actual data path
    desired_runs = ('04', '08', '12')
    # Event dictionary: mapping annotation keys to numeric values
    event_dict = {'T0': 1, 'T1': 2, 'T2': 3}
    # Epoch parameters
    tmin, tmax, baseline = -0.2, 3.8, (None, 0)
    # Sampling frequency (adjust as needed)
    sfreq = 160
    # FBCSP parameters
    freq_bands = [(8, 12), (12, 16), (16, 20)]  # example frequency bands
    n_csp = 4
    # Classifier parameters (for MLPClassifier)
    classifier_params = {
        'hidden_layer_sizes': (100,),
        'max_iter': 300,
        'random_state': 42
    }

    # 2. Load file information
    df_files = load_file_info(data_directory, desired_runs=desired_runs)

    # 3. Process and load all EEG data and corresponding labels
    X_all, y_all = load_all_data(df_files, event_dict)

    # 4. Build the machine learning pipeline
    pipeline = build_pipeline(freq_bands, n_csp, sfreq, classifier_params)

    # 5. Fit the pipeline on the entire dataset
    logging.info("Fitting the pipeline on the entire dataset...")
    pipeline.fit(X_all, y_all)

    # 6. Evaluate the pipeline
    evaluate_pipeline(pipeline, X_all, y_all)


if __name__ == '__main__':
    main()