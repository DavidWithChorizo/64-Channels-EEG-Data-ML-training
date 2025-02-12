#!/usr/bin/env python
"""
EEGNet Optuna Hyperparameter Optimization Pipeline for EEG Motor Imagery Data

This script loads and preprocesses EEG motor imagery data from EDF files,
selects a subset of channels (e.g., 8 out of 64) that resemble conventional EEG headset channels,
builds an EEGNet model using TensorFlow/Keras, and performs hyperparameter tuning using Optuna.
A progress bar (using tqdm) is included to track the optimization process.

The EEGNet architecture implemented here is adapted from:
Lawhern, V. J., et al. "EEGNet: A Compact Convolutional Neural Network for EEG-based
Brain–Computer Interfaces." Journal of Neural Engineering, 2018.

Usage:
    python optuna_eegnet_pipeline.py
"""

import os
import re
import logging
from pathlib import Path
import tensorflow as tf
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

import optuna
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

###############################################################################
# DATA LOADING AND PREPROCESSING FUNCTIONS
###############################################################################
def load_file_info(data_dir, desired_runs=('04', '08', '12')):
    """
    Load information about EDF files from the given data directory.

    Parameters
    ----------
    data_dir : str or Path
        Path to the directory containing subject folders with EDF files.
    desired_runs : tuple of str, optional
        EDF files corresponding to these run identifiers will be matched.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing file information for each matched EDF file.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"The data directory does not exist: {data_dir}")
    # Regular expression to match EDF files from desired runs (e.g., '04', '08', '12')
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
    Process a single EDF file: load raw data, extract events and epochs,
    and return EEG data and labels.

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing file information.
    event_dict : dict
        Dictionary mapping event names to event codes.
    tmin : float, optional
        Start time before the event.
    tmax : float, optional
        End time after the event.
    baseline : tuple, optional
        Baseline correction period.

    Returns
    -------
    tuple or (None, None)
        Tuple containing the EEG data (ndarray) of shape (n_epochs, n_channels, n_times)
        and labels (ndarray). Returns (None, None) if processing fails.
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
        # Select only motor imagery epochs (e.g., T1 and T2)
        motor_epochs = epochs[['T1', 'T2']]
        if len(motor_epochs) == 0:
            logging.warning(f"No motor-related epochs found in {edf_file}. Skipping.")
            return None, None
        X = motor_epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        y = motor_epochs.events[:, -1] - event_dict['T1']  # Adjust labels to be 0-indexed
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
        Start time before the event.
    tmax : float, optional
        End time after the event.
    baseline : tuple, optional
        Baseline correction period.

    Returns
    -------
    tuple
        Tuple containing concatenated EEG data (ndarray) and labels (ndarray).
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

    Converts data from shape (n_trials, n_channels, n_times) to
    (n_trials, len(selected_channels), n_times) by selecting specific channels.
    
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
        # Check if X is 3D: (n_trials, n_channels, n_times)
        if X.ndim != 3:
            raise ValueError(f"Expected X to be 3D, got shape: {X.shape}")
        return X[:, self.channels, :]

class ExpandDimsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to expand EEG data dimensions for compatibility with CNNs.

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
def build_eegnet_model(dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
                       input_shape=(8, 641, 1), nb_classes=2):
    """
    Build and compile an EEGNet model.

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
        Shape of the input data (n_channels, n_times, 1).
    nb_classes : int, optional
        Number of output classes.

    Returns
    -------
    tf.keras.Model
        Compiled EEGNet model.
    """
    input1 = Input(shape=input_shape)
    # First block: Temporal convolution and spatial filtering
    x = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((input_shape[0], 1), use_bias=False,
                        depth_multiplier=D,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropoutRate)(x)
    # Second block: Separable convolution
    x = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropoutRate)(x)
    # Classification block
    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(0.5))(x)
    output = Activation('softmax')(x)
    # Create and compile model
    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

###############################################################################
# OPTUNA OBJECTIVE FUNCTION
###############################################################################
def objective(trial, X_train, y_train, X_val, y_val, input_shape, nb_classes, selected_channels):
    """
    Objective function for Optuna hyperparameter optimization.

    This function builds an EEGNet model with hyperparameters suggested by the trial,
    trains the model on the training data, and evaluates it on the validation data.

    Parameters
    ----------
    trial : optuna.trial.Trial
        An Optuna trial object for hyperparameter suggestions.
    X_train : ndarray
        Training data of shape (n_trials, n_channels, n_times).
    y_train : ndarray
        Training labels.
    X_val : ndarray
        Validation data.
    y_val : ndarray
        Validation labels.
    input_shape : tuple
        Shape of the input data for the CNN (n_selected_channels, n_times, 1).
    nb_classes : int
        Number of output classes.
    selected_channels : list of int
        List of channel indices to select.
    
    Returns
    -------
    float
        Validation accuracy of the trained model.
    """
    # Suggest hyperparameters from predefined search spaces
    dropoutRate = trial.suggest_float("dropoutRate", 0.25, 0.5, step=0.05)
    kernLength = trial.suggest_categorical("kernLength", [32, 64])
    F1 = trial.suggest_categorical("F1", [8, 16])
    D = trial.suggest_int("D", 1, 2)
    F2 = trial.suggest_categorical("F2", [16, 32])
    epochs = trial.suggest_int("epochs", 50, 100, step=50)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    # Build the EEGNet model with the current hyperparameters
    model = build_eegnet_model(dropoutRate=dropoutRate,
                               kernLength=kernLength,
                               F1=F1,
                               D=D,
                               F2=F2,
                               input_shape=input_shape,
                               nb_classes=nb_classes)

    # Wrap the model in a KerasClassifier for compatibility with scikit-learn
    clf = KerasClassifier(
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    # Create a pipeline that selects channels, expands dimensions, then fits the EEGNet classifier
    pipeline = Pipeline([
        ('channel_selector', ChannelSelector(selected_channels)),
        ('expand_dims', ExpandDimsTransformer()),
        ('eegnet', clf)
    ])

    # Fit the pipeline on the training data and evaluate on the validation set
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_val, y_val)
    logging.info(f"Trial {trial.number}: Val Accuracy = {score * 100:.2f}%")
    return score

###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    """
    Main function to run the EEGNet hyperparameter optimization using Optuna.
    
    Loads EEG data, selects a subset of channels (e.g., 8 out of 64), splits data into training, 
    validation, and test sets, and runs Optuna to optimize EEGNet hyperparameters. A tqdm progress bar 
    is displayed to track the optimization process.
    """
    # Set parameters for data loading
    data_directory = './data/raw data'  # Adjust this path as needed
    desired_runs = ('04', '08', '12')
    event_dict = {'T0': 1, 'T1': 2, 'T2': 3}
    tmin, tmax, baseline = -0.2, 3.8, (None, 0)

    # Check if GPUs are available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"GPUs available: {gpus}")
    else:
        logging.info("No GPUs found. Running on CPU.")

    # Define the selected channels (example: 8 channels evenly spaced across 64 channels)
    # Adjust these indices as needed to match conventional EEG headset channels.
    selected_channels = [0, 8, 16, 24, 32, 40, 48, 56]

    # Load file information and EEG data
    df_files = load_file_info(data_directory, desired_runs=desired_runs)
    if df_files.empty:
        raise RuntimeError("No matching EDF files found.")
    X_all, y_all = load_all_data(df_files, event_dict, tmin=tmin, tmax=tmax, baseline=baseline)

    # Split data into training+validation and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    # Further split training data into training and validation sets for hyperparameter tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    logging.info(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    # Before channel selection, X_train has shape (n_trials, 64, n_times).
    # Compute the new input shape after channel selection and expansion.
    # For example, if X_train[0] has shape (64, 641), then after channel selection,
    # the shape becomes (8, 641). After expanding dimensions, it becomes (8, 641, 1).
    sample = X_train[0]  # shape: (64, n_times)
    # Get the new number of channels
    new_n_channels = len(selected_channels)
    # Get the number of time samples (assumed consistent across channels)
    n_times = sample.shape[1]
    input_shape = (new_n_channels, n_times, 1)
    logging.info(f"New input shape for EEGNet after channel selection: {input_shape}, Number of classes: {len(np.unique(y_all))}")

    # Create an Optuna study to maximize validation accuracy
    study = optuna.create_study(direction="maximize")
    n_trials = 30  # Total number of trials for optimization

    # Set up a progress bar using tqdm
    progress_bar = tqdm(total=n_trials, desc="Optuna Trials")

    # Callback function to update the progress bar after each trial
    def progress_callback(study, trial):
        progress_bar.update(1)

    # Optimize the objective function using Optuna, passing in the selected channels and new input shape.
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val,
                                            input_shape, len(np.unique(y_all)), selected_channels),
                   n_trials=n_trials,
                   callbacks=[progress_callback])
    progress_bar.close()

    # Log the best hyperparameters and corresponding validation accuracy
    logging.info(f"Best Hyperparameters: {study.best_params}")
    logging.info(f"Best Validation Accuracy: {study.best_value * 100:.2f}%")

    # Retrain the best model on the full training+validation set with channel selection in the pipeline
    best_params = study.best_params
    final_model = build_eegnet_model(
        dropoutRate=best_params["dropoutRate"],
        kernLength=best_params["kernLength"],
        F1=best_params["F1"],
        D=best_params["D"],
        F2=best_params["F2"],
        input_shape=input_shape,
        nb_classes=len(np.unique(y_all))
    )
    final_clf = KerasClassifier(
        model=final_model,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        verbose=0
    )
    final_pipeline = Pipeline([
        ('channel_selector', ChannelSelector(selected_channels)),
        ('expand_dims', ExpandDimsTransformer()),
        ('eegnet', final_clf)
    ])
    # Fit on the full training+validation set and evaluate on the held-out test set
    final_pipeline.fit(X_train_full, y_train_full)
    test_accuracy = final_pipeline.score(X_test, y_test)
    logging.info(f"Test Accuracy of the final model: {test_accuracy * 100:.2f}%")
    print("Final Test Accuracy: {:.2f}%".format(test_accuracy * 100))

if __name__ == '__main__':
    main()