# %%
import numpy as np
import mne
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier  # Neural Network Classifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mne.filter import filter_data
import matplotlib.pyplot as plt
# Import the required Python libraries
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import mne

# %%
# Define the path to your raw_data directory
data_dir = Path('../data/raw data')  # Adjust this path as needed

# Verify the directory exists
if not data_dir.exists():
    raise FileNotFoundError(f"The specified data directory does not exist: {data_dir}")

# Define the specific runs to process, including both two-digit and three-digit run numbers
desired_runs = ['04', '08', '12']

# Compile a regular expression pattern to match filenames like S001R04.edf, S002R08.edf, S001R012.edf, etc.
# This pattern ensures that only R04, R08, R12 are matched
edf_pattern = re.compile(r'^S\d{3}R0?(04|08|12)\.edf$', re.IGNORECASE)

# %%
# Initialize a list to store file information
file_info = []

# Counter for processed subject directories
processed_subjects = 0

# Iterate through each subject folder in raw_data
for subject_dir in data_dir.iterdir():
    if subject_dir.is_dir() and re.match(r'^S\d{3}$', subject_dir.name, re.IGNORECASE):
        processed_subjects += 1
        # Iterate through each file in the subject directory
        for file in subject_dir.iterdir():
            if file.is_file() and edf_pattern.match(file.name):
                # Derive the corresponding event file name
                base_name = file.stem  # e.g., S001R04
                event_file = f"{base_name}.edf.event"
                event_path = subject_dir / event_file
                
                # Check if the event file exists
                event_exists = event_path.exists()
                
                # Append the information to the list
                file_info.append({
                    'subject': subject_dir.name,
                    'edf_file': file.name,
                    'event_file': event_file,
                    'event_exists': event_exists,
                    'edf_path': str(file.resolve()),
                    'event_path': str(event_path.resolve()) if event_exists else None
                })

# Convert the list to a DataFrame for better visualization
df_files = pd.DataFrame(file_info)

# Display summary of processed subject directories
print(f"Total subject directories processed: {processed_subjects}")

# %%
# Total number of matched .edf files
total_edf = df_files.shape[0]

# Number of event files present
total_events = df_files['event_exists'].sum()

# Number of missing event files
missing_events = total_edf - total_events

print(f"Total matched .edf files: {total_edf}")
print(f"Number of corresponding event files found: {total_events}")
print(f"Number of missing event files: {missing_events}")

# %%
# Filter the DataFrame for missing event files
df_missing_events = df_files[~df_files['event_exists']]

# Display the list of .edf files without corresponding event files
if not df_missing_events.empty:
    print("\nFiles missing corresponding event files:")
    display(df_missing_events[['subject', 'edf_file', 'event_file']])
else:
    print("\nAll matched .edf files have corresponding event files.")

# %%
# Display the entire DataFrame of matched files
print("List of All Matched .edf Files:")
display(df_files[['subject', 'edf_file', 'event_file', 'event_exists']])

# %%
# Initialize lists to collect data and labels from all files
all_epochs = []
all_labels = []

# Iterate through each matched EDF file
for idx, row in df_files.iterrows():
    edf_path = row['edf_path']
    event_path = row['event_path']
    subject = row['subject']
    edf_file = row['edf_file']
    
    print(f"\nProcessing file: {edf_file} for subject: {subject}")
    
    try:
        # Load the raw EEG data
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Access annotations
        annotations = raw.annotations
        
        # Display annotations (optional)
        print(f"Annotations for {edf_file}:")
        print(annotations)
        
        # Define event dictionary
        event_dict = {
            'Rest': 1,                      # T0
            'Imagining Left Hand': 2,      # T1
            'Imagining Right Hand': 3      # T2
        }
        
        # Convert annotations to events and event_id
        events, event_id = mne.events_from_annotations(raw, event_id=event_dict)
        
        # Update event_id with descriptive labels (if necessary)
        # event_id = {
        #     'Rest': 1,
        #     'Imagining Left Hand': 2,
        #     'Imagining Right Hand': 3
        # }
        
        # Define epoch parameters
        tmin = -0.2  # 200 ms before the event
        tmax = 3.8   # 3800 ms after the event
        baseline = (None, 0)  # Baseline correction
        
        # Create epochs
        epochs = Epochs(raw, events, event_id=event_dict, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, verbose=False)
        
        # Select only 'Imagining Left Hand' and 'Imagining Right Hand' epochs
        motor_epochs = epochs[['Imagining Left Hand', 'Imagining Right Hand']]
        
        # Extract labels: 0 for Left, 1 for Right
        labels = motor_epochs.events[:, -1] - 2  # 'Imagining Left Hand' = 2, 'Imagining Right Hand' = 3
        
        # Get the data as a NumPy array: shape (n_epochs, n_channels, n_times)
        X = motor_epochs.get_data()
        y = labels
        
        # Append to the aggregate lists
        all_epochs.append(X)
        all_labels.append(y)
        
        print(f"Loaded {X.shape[0]} epochs from {edf_file}.")
        
    except Exception as e:
        print(f"Failed to process {edf_file}. Error: {e}")

# Concatenate all epochs and labels
if all_epochs and all_labels:
    X_all = np.concatenate(all_epochs, axis=0)
    y_all = np.concatenate(all_labels, axis=0)
    
    print(f"\nTotal epochs collected: {X_all.shape[0]}")
    print(f"Shape of X_all: {X_all.shape}")
    print(f"Shape of y_all: {y_all.shape}")
else:
    raise ValueError("No epochs were loaded. Please check your EDF files and event annotations.")

# %%
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

# %%
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
            X_features = np.column_stack(features)
            return X_features

# %%
# Determine the sampling frequency from one of the loaded raw files
# Assuming all files have the same sampling frequency
# If not, you'll need to handle varying sampling frequencies appropriately
example_raw = mne.io.read_raw_edf(df_files.iloc[0]['edf_path'], preload=False, verbose=False)
sfreq = example_raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")

# %%
# Initialize FBCSP transformer with the determined sampling frequency
fbcsp = FBCSP(freq_bands=freq_bands, n_csp=4, sfreq=sfreq)

# Initialize Neural Network classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# Define the classification pipeline
pipeline = Pipeline([
    ('fbcsp', fbcsp),
    ('classifier', mlp)
])

# %%
# Define cross-validation strategy
n_splits = 5  # Adjust based on your dataset size
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Perform cross-validation
print("Starting Cross-Validation...")
scores = cross_val_score(pipeline, X_all, y_all, cv=cv, scoring='accuracy')

print(f"\nCross-Validation Accuracy Scores: {scores}")
print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {scores.std() * 100:.2f}%")

# %%
# Optional: Generate a classification report and confusion matrix
# Perform cross-validated predictions
print("\nGenerating Classification Report and Confusion Matrix...")
y_pred = cross_val_predict(pipeline, X_all, y_all, cv=cv)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_all, y_pred, target_names=['Imagining Left Hand', 'Imagining Right Hand']))

# Confusion Matrix
conf_mat = confusion_matrix(y_all, y_pred)
print("\nConfusion Matrix:")
print(conf_mat)

# Optional: Plot the Confusion Matrix
import seaborn as sns

plt.figure(figsize=(6,5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Imagining Left Hand', 'Imagining Right Hand'],
            yticklabels=['Imagining Left Hand', 'Imagining Right Hand'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()