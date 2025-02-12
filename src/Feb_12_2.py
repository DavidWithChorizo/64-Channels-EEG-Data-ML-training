#!/usr/bin/env python
"""
EEGNet Training and Evaluation Pipeline (Fixed Hyperparameters)

This script loads and preprocesses EEG motor imagery data from EDF files,
selects a subset of channels (8 out of 64, based on predefined indices that resemble
a conventional EEG headset configuration), and trains an EEGNet model using fixed
hyperparameters (derived from a previous hyperparameter tuning trial).

The EEGNet model is defined using the fixed hyperparameters and compiled using the
        "dropoutRate": 0.3,
        "kernLength": 32,
        "F1": 8,
        "D": 1,
        "F2": 16,
        "epochs": 100,
        "batch_size": 32


The model is trained using an 80/20 train/test split.

Usage:
    python fixed_hyperparams_eegnet.py
"""

import os
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from mne import Epochs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, DepthwiseConv2D, Activation, \
    AveragePooling2D, Dropout, SeparableConv2D, Flatten, Dense
from tensorflow.keras.constraints import max_norm

from scikeras.wrappers import KerasClassifier

# Configure logging to display information during execution
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

###############################################################################
# DATA LOADING AND PREPROCESSING FUNCTIONS
###############################################################################
def load_file_info(data_dir, desired_runs=('04', '08', '12')):
    """
    Load information about EDF files from the specified data directory.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing subject folders with EDF files.
    desired_runs : tuple of str, optional
        Only EDF files corresponding to these run identifiers are matched.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing file information (subject, filename, and file path)
        for each matched EDF file.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"The data directory does not exist: {data_dir}")
    # Regular expression to match EDF files (e.g., SXXXR04.edf, SXXXR08.edf, SXXXR12.edf)
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
    Process a single EDF file: load raw EEG data, extract events and epochs,
    and return the EEG epochs and labels.

    Parameters
    ----------
    row : pd.Series
        A row from the file information DataFrame.
    event_dict : dict
        Dictionary mapping event names to event codes.
    tmin : float, optional
        Start time (in seconds) relative to each event.
    tmax : float, optional
        End time (in seconds) relative to each event.
    baseline : tuple, optional
        The time interval for baseline correction.

    Returns
    -------
    tuple or (None, None)
        A tuple (X, y) where X is an ndarray of shape (n_epochs, n_channels, n_times)
        and y is an ndarray of labels. Returns (None, None) if processing fails.
    """
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
        # Select only the motor imagery epochs (e.g., events 'T1' and 'T2')
        motor_epochs = epochs[['T1', 'T2']]
        if len(motor_epochs) == 0:
            logging.warning(f"No motor-related epochs found in {edf_file}. Skipping.")
            return None, None
        X = motor_epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        # Adjust labels so that they start from 0
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
    """
    Load and concatenate EEG data and labels from multiple EDF files.

    Parameters
    ----------
    df_files : pd.DataFrame
        DataFrame containing EDF file information.
    event_dict : dict
        Dictionary mapping event names to event codes.
    tmin : float, optional
        Start time (in seconds) relative to each event.
    tmax : float, optional
        End time (in seconds) relative to each event.
    baseline : tuple, optional
        The time interval for baseline correction.

    Returns
    -------
    tuple
        A tuple (X_all, y_all) where X_all is an ndarray of shape 
        (total_n_epochs, n_channels, n_times) and y_all is an ndarray of labels.
    """
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
# CUSTOM TRANSFORMERS
###############################################################################
class ChannelSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a subset of channels from EEG data.

    Reduces data shape from (n_trials, n_channels, n_times) to
    (n_trials, len(selected_channels), n_times) by selecting specific channel indices.
    
    Parameters
    ----------
    channels : list of int
        List of channel indices to select.
    """
    def __init__(self, channels):
        self.channels = channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X has three dimensions
        if X.ndim != 3:
            raise ValueError(f"Expected X to be 3D, got shape: {X.shape}")
        return X[:, self.channels, :]

class ExpandDimsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to add an extra dimension to EEG data for compatibility with CNNs.

    Converts data from shape (n_trials, n_channels, n_times) to
    (n_trials, n_channels, n_times, 1).
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.ndim != 3:
            raise ValueError(f"Expected X to be 3D (n_trials, n_channels, n_times), got shape: {X.shape}")
        return X[..., np.newaxis]

###############################################################################
# EEGNET MODEL DEFINITION
###############################################################################
def build_eegnet_model(dropoutRate=0.3, kernLength=32, F1=8, D=1, F2=16,
                       input_shape=(8, 641, 1), nb_classes=2):
    """
    Build and compile an EEGNet model with fixed hyperparameters.

    Parameters
    ----------
    dropoutRate : float, optional
        Dropout rate for regularization.
    kernLength : int, optional
        Length of the temporal convolution kernel.
    F1 : int, optional
        Number of temporal filters.
    D : int, optional
        Depth multiplier for depthwise convolution.
    F2 : int, optional
        Number of pointwise filters in separable convolution.
    input_shape : tuple, optional
        Shape of the input data (n_selected_channels, n_times, 1).
    nb_classes : int, optional
        Number of output classes.

    Returns
    -------
    tf.keras.Model
        A compiled EEGNet model.
    """
    input1 = Input(shape=input_shape)
    
    # First convolution block: Temporal convolution + depthwise convolution
    x = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((input_shape[0], 1), use_bias=False,
                        depth_multiplier=D,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropoutRate)(x)
    
    # Second convolution block: Separable convolution
    x = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropoutRate)(x)
    
    # Classification block
    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(0.5))(x)
    output = Activation('softmax')(x)
    
    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    """
    Main function to load EEG data, build the model using fixed hyperparameters,
    train the model on an 80/20 train/test split, and evaluate performance on the test set.
    """
    # Data loading parameters
    data_directory = './data/raw data'  # Adjust this path if needed
    desired_runs = ('04', '08', '12')
    event_dict = {'T0': 1, 'T1': 2, 'T2': 3}
    tmin, tmax, baseline = -0.2, 3.8, (None, 0)

    # Check for available GPUs (for informational purposes)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"GPUs available: {gpus}")
    else:
        logging.info("No GPUs found. Running on CPU.")

    # Define selected channel indices (for example, 8 channels evenly spaced in a 64-channel montage)
    selected_channels = [9,13,22,24,30,38,61,63]

    # Load file information and data
    df_files = load_file_info(data_directory, desired_runs=desired_runs)
    if df_files.empty:
        raise RuntimeError("No matching EDF files found.")
    X_all, y_all = load_all_data(df_files, event_dict, tmin=tmin, tmax=tmax, baseline=baseline)

    # Split the data into train+validation and test sets (80/20 split)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    logging.info(f"Train+Validation set shape: {X_train_full.shape}, Test set shape: {X_test.shape}")

    # Compute new input shape after channel selection:
    # Original data shape is (n_trials, 64, n_times). After channel selection, it becomes
    # (n_trials, 8, n_times) and then after expanding dimensions: (n_trials, 8, n_times, 1).
    sample = X_train_full[0]  # shape: (64, n_times)
    new_n_channels = len(selected_channels)
    n_times = sample.shape[1]
    input_shape = (new_n_channels, n_times, 1)
    logging.info(f"New input shape for EEGNet: {input_shape}, Number of classes: {len(np.unique(y_all))}")

    # Define fixed hyperparameters for the EEGNet model
    fixed_params = {
        "dropoutRate": 0.3,
        "kernLength": 32,
        "F1": 8,
        "D": 1,
        "F2": 16,
        "epochs": 100,
        "batch_size": 32
    }
    
    # Build the EEGNet model with the fixed hyperparameters
    model = build_eegnet_model(dropoutRate=fixed_params["dropoutRate"],
                               kernLength=fixed_params["kernLength"],
                               F1=fixed_params["F1"],
                               D=fixed_params["D"],
                               F2=fixed_params["F2"],
                               input_shape=input_shape,
                               nb_classes=len(np.unique(y_all)))
    
    # Wrap the model with scikeras's KerasClassifier
    clf = KerasClassifier(
        model=model,
        epochs=fixed_params["epochs"],
        batch_size=fixed_params["batch_size"],
        verbose=1
    )
    
    # Create a pipeline that applies channel selection, dimension expansion, and the classifier
    pipeline = Pipeline([
        ('channel_selector', ChannelSelector(selected_channels)),
        ('expand_dims', ExpandDimsTransformer()),
        ('eegnet', clf)
    ])
    
    # Train the model on the full training set (80% of the data)
    logging.info("Training the model with fixed hyperparameters...")
    pipeline.fit(X_train_full, y_train_full)
    
    # Evaluate the final model on the test set (20% of the data)
    test_accuracy = pipeline.score(X_test, y_test)
    logging.info(f"Test Accuracy of the final model: {test_accuracy * 100:.2f}%")
    print("Final Test Accuracy: {:.2f}%".format(test_accuracy * 100))

    # Save the final model to disk
    if test_accuracy > 0.7956:
        logging.info("Saving the final model to disk...")
        final_model = pipeline.named_steps['eegnet'].model_
        final_model.save("final_eegnet_model.h5")
        logging.info("Model saved to final_eegnet_model.h5")



if __name__ == '__main__':
    main()