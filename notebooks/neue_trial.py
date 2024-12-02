import numpy as np
import mne
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier  # Neural Network Classifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mne.filter import filter_data
import matplotlib.pyplot as plt

# ===========================
# 1. Data Loading and Preprocessing
# ===========================

# Define the path to the EDF file
edf_file = '../data/raw data/S001/S001R04.edf'

# Load the raw EEG data with annotations
raw = mne.io.read_raw_edf(edf_file, preload=True)

# Access annotations
annotations = raw.annotations

# Display annotations
print(annotations)

# Define event dictionary
event_dict = {
    'Rest': 1,                # T0
    'Imagining Left Hand': 2,  # T1
    'Imagining Right Hand': 3  # T2
}

# Convert annotations to events and event_id
events, event_id = mne.events_from_annotations(raw)

# Update event_id with descriptive labels
event_id = {
    'Rest': 1,
    'Imagining Left Hand': 2,
    'Imagining Right Hand': 3
}

print(event_id)

from mne import Epochs

# Define epoch parameters
tmin = -0.2  # 200 ms before the event
tmax = 3.8   # 3800 ms after the event
baseline = (None, 0)  # Baseline correction

# Create epochs
epochs = Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=baseline, preload=True)

# Display epoch information
print(epochs)

# Select only 'Imagining Left Hand' and 'Imagining Right Hand' epochs
motor_epochs = epochs[['Imagining Left Hand', 'Imagining Right Hand']]

# Extract labels: 0 for Left, 1 for Right
labels = motor_epochs.events[:, -1] - 2  # 'Imagining Left Hand' = 2, 'Imagining Right Hand' = 3

# Verify the distribution of classes
print("Class distribution:", np.bincount(labels))

# Get the data as a NumPy array: shape (n_epochs, n_channels, n_times)
X = motor_epochs.get_data()
y = labels

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# ===========================
# 2. Define Frequency Bands
# ===========================

# Define frequency bands (in Hz)
freq_bands = [
    (8, 12),    # Alpha
    (13, 20),   # Beta 1
    (21, 30),   # Beta 2
    (31, 40),   # Low Gamma
    (41, 50),   # High Gamma
    (51, 60),   # Very High Gamma
    (61, 70),   # Ultra High Gamma
    (71, 79)    # Max Gamma
]

# ===========================
# 3. Define the Custom FBCSP Transformer
# ===========================

class FBCSP(BaseEstimator, TransformerMixin):
    def __init__(self, freq_bands, n_csp=4, sfreq=160):
        """
        Initialize the FBCSP transformer.
        
        Parameters:
        - freq_bands: List of tuples defining frequency bands.
        - n_csp: Number of CSP components per band.
        - sfreq: Sampling frequency of the EEG data.
        """
        self.freq_bands = freq_bands
        self.n_csp = n_csp
        self.sfreq = sfreq
        self.csp_list_ = []  # To store CSP models for each band

    def fit(self, X, y):
        """
        Fit CSP models for each frequency band.
        
        Parameters:
        - X: EEG data, shape (n_trials, n_channels, n_times)
        - y: Labels, shape (n_trials,)
        
        Returns:
        - self
        """
        self.csp_list_ = []  # Reset in case of multiple fits
        for band in self.freq_bands:
            print(f'Fitting CSP for band: {band[0]}-{band[1]} Hz')
            # Band-pass filter the data for the current frequency band
            X_filtered = np.array([
                filter_data(trial, sfreq=self.sfreq, l_freq=band[0], h_freq=band[1], method='iir', verbose=False)
                for trial in X
            ])
            
            # Initialize and fit CSP
            csp = CSP(n_components=self.n_csp, reg=None, log=True, norm_trace=False)
            csp.fit(X_filtered, y)
            self.csp_list_.append(csp)
        return self

    def transform(self, X):
        """
        Transform EEG data using the fitted CSP models.
        
        Parameters:
        - X: EEG data, shape (n_trials, n_channels, n_times)
        
        Returns:
        - X_features: Extracted features, shape (n_trials, n_bands * n_csp)
        """
        features = []
        for idx, band in enumerate(self.freq_bands):
            print(f'Transforming data for band: {band[0]}-{band[1]} Hz')
            csp = self.csp_list_[idx]
            # Band-pass filter the data for the current frequency band
            X_filtered = np.array([
                filter_data(trial, sfreq=self.sfreq, l_freq=band[0], h_freq=band[1], method='iir', verbose=False)
                for trial in X
            ])
            
            # Apply CSP transformation
            X_csp = csp.transform(X_filtered)
            
            # Extract log-variance features
            X_logvar = np.log(np.var(X_csp, axis=1))
            
            # Reshape X_logvar to (n_trials, 1) to make it 2D
            X_logvar = X_logvar.reshape(-1, 1)
            
            # Append the reshaped feature
            features.append(X_logvar)
        
        # Concatenate features from all bands along axis=1
        X_features = np.concatenate(features, axis=1)
        return X_features

# ===========================
# 4. Define the Classification Pipeline
# ===========================

# Initialize FBCSP transformer
fbcsp = FBCSP(freq_bands=freq_bands, n_csp=4, sfreq=160)  # Replace sfreq with your actual sampling frequency

# Initialize Neural Network classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# Define the classification pipeline
pipeline = Pipeline([
    ('fbcsp', fbcsp),
    ('classifier', mlp)
])

# ===========================
# 5. Evaluate the Pipeline with Cross-Validation
# ===========================

# Define cross-validation strategy
n_splits = 5  # Adjust based on your dataset size
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Perform cross-validation
print("Starting Cross-Validation...")
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

print(f"\nCross-Validation Accuracy Scores: {scores}")
print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {scores.std() * 100:.2f}%")

# ===========================
# 6. Detailed Evaluation with Predictions
# ===========================

# Obtain cross-validated predictions
print("\nGenerating Cross-Validated Predictions...")
y_pred = cross_val_predict(pipeline, X, y, cv=cv)

# Compute and display classification metrics
print("\nClassification Report:")
print(classification_report(y, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

# ===========================
# 7. Hyperparameter Tuning with GridSearchCV (Optional)
# ===========================

# Define parameter grid
param_grid = {
    'fbcsp__n_csp': [2, 4, 6],  # Number of CSP components per band
    'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'classifier__alpha': [0.0001, 0.001, 0.01]
}

# Initialize GridSearchCV
print("\nStarting Grid Search for Hyperparameter Tuning...")
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy')
grid.fit(X, y)

# Best parameters and score
print("\nBest Parameters from GridSearchCV:")
print(grid.best_params_)
print(f"Best Cross-Validation Accuracy: {grid.best_score_ * 100:.2f}%")