#!/usr/bin/env python
"""
EEGNet Classification Pipeline for EEG Motor Imagery Data

This script loads and preprocesses EEG motor imagery data from EDF files,
builds an EEGNet model using TensorFlow/Keras, and performs hyperparameter
tuning using GridSearchCV from scikit-learn. The final model is then evaluated
on a held-out test set.

The EEGNet architecture implemented here is adapted from the original EEGNet paper:
Lawhern, V. J., et al. "EEGNet: A Compact Convolutional Neural Network for EEG-based
Brain–Computer Interfaces." Journal of Neural Engineering, 2018.

Usage:
    python eegnet_pipeline.py
"""

import os
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from mne import Epochs
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, DepthwiseConv2D, Activation, AveragePooling2D, Dropout, SeparableConv2D, Flatten, Dense
from tensorflow.keras.constraints import max_norm
from scikeras.wrappers import KerasClassifier

print("All modules imported successfully!")

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        # Optional: Check for expected number of time samples (here expected to be 641)
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
class ExpandDimsTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to expand the dimensions of EEG data to add a channel axis.
    
    Converts data from shape (n_trials, n_channels, n_times) to
    (n_trials, n_channels, n_times, 1) for compatibility with CNNs.
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
                       input_shape=(64, 128, 1), nb_classes=2):
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
    # Input layer
    input1 = Input(shape=input_shape)
    
    # First convolution block (temporal convolution)
    x = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    x = BatchNormalization()(x)
    # Depthwise convolution: learns spatial filters across channels
    x = DepthwiseConv2D((input_shape[0], 1), use_bias=False,
                        depth_multiplier=D,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropoutRate)(x)
    
    # Separable convolution block (combining depthwise and pointwise convolutions)
    x = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropoutRate)(x)
    
    # Classification block
    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(0.5))(x)
    output = Activation('softmax')(x)
    
    # Create and compile the model
    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

###############################################################################
# MAIN PIPELINE AND TRAINING FUNCTION
###############################################################################
def main():
    """
    Main function to run the EEGNet classification pipeline.
    
    Loads data, preprocesses it, builds an EEGNet model, performs hyperparameter tuning,
    and evaluates the final model on a held-out test set.
    """
    # Set data directory and parameters for loading EDF files
    data_directory = './data/raw data'  # Adjust to your actual data path
    desired_runs = ('04', '08', '12')
    event_dict = {'T0': 1, 'T1': 2, 'T2': 3}
    tmin, tmax, baseline = -0.2, 3.8, (None, 0)
    
    # Load file information and EEG data
    df_files = load_file_info(data_directory, desired_runs=desired_runs)
    if df_files.empty:
        raise RuntimeError("No matching EDF files found.")
    X_all, y_all = load_all_data(df_files, event_dict, tmin=tmin, tmax=tmax, baseline=baseline)
    
    # At this point, X_all has shape (n_trials, n_channels, n_times)
    # We will expand the dimensions (i.e., add a channel axis) via our custom transformer in the pipeline.
    
    # Split data into training+validation and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    logging.info(f"Training+Validation set shape: {X_train_full.shape}, Test set shape: {X_test.shape}")
    
    # Further split training data into training and validation sets for hyperparameter tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    logging.info(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
    
    # Determine input shape for the CNN by expanding a sample manually.
    sample_expanded = np.expand_dims(X_train[0], axis=-1)
    input_shape = sample_expanded.shape  # Expected shape: (n_channels, n_times, 1)
    nb_classes = len(np.unique(y_all))
    logging.info(f"Input shape for EEGNet: {input_shape}, Number of classes: {nb_classes}")
    
    # Build a scikit-learn pipeline for EEGNet classification.
    # The pipeline includes:
    # 1. 'expand_dims': converts data from (n_trials, n_channels, n_times) to (n_trials, n_channels, n_times, 1)
    # 2. 'eegnet': a KerasClassifier wrapper around our EEGNet model.
    eegnet_pipeline = Pipeline([
        ('expand_dims', ExpandDimsTransformer()),
        ('eegnet', KerasClassifier(
            model=build_eegnet_model,
            # Initial model parameters; these will be overridden during hyperparameter tuning.
            model__input_shape=input_shape,
            model__nb_classes=nb_classes,
            epochs=50,
            batch_size=16,
            verbose=0
        ))
    ])
    
    # Define a grid of hyperparameters to search over.
    # The keys use the "step__parameter" naming convention for pipeline parameters.
    param_grid = {
        'eegnet__model__dropoutRate': [0.5, 0.25],
        'eegnet__model__kernLength': [32, 64],
        'eegnet__model__F1': [8, 16],
        'eegnet__model__D': [1, 2],
        'eegnet__model__F2': [16, 32],
        'eegnet__epochs': [50, 100],
        'eegnet__batch_size': [16, 32]
    }
    
    # Perform hyperparameter tuning using GridSearchCV on the training set
    logging.info("Starting hyperparameter tuning using GridSearchCV...")
    grid_search = GridSearchCV(estimator=eegnet_pipeline,
                               param_grid=param_grid,
                               cv=3,
                               scoring='accuracy',
                               verbose=2)
    
    # Fit grid search on the training data
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Best hyperparameters found: {grid_search.best_params_}")
    logging.info(f"Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%")
    
    # Evaluate the best model on the validation set
    best_model = grid_search.best_estimator_
    val_accuracy = best_model.score(X_val, y_val)
    logging.info(f"Validation accuracy of the best model: {val_accuracy * 100:.2f}%")
    
    # Retrain the best model on the full training+validation set
    logging.info("Retraining best model on the full training+validation set...")
    best_model.fit(X_train_full, y_train_full)
    
    # Evaluate the final model on the held-out test set
    test_accuracy = best_model.score(X_test, y_test)
    logging.info(f"Test accuracy of the final model: {test_accuracy * 100:.2f}%")
    
    # Print final results
    print("Final Test Accuracy: {:.2f}%".format(test_accuracy * 100))

if __name__ == '__main__':
    main()