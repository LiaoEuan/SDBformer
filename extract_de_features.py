# =============================================================================
# Feature Extraction: Differential Entropy (DE) for EEG Signals
# Description: Extracts DE features from raw EEG slices across 5 frequency bands.
# Expected Input: .npz files containing 'eeg_slices' (N, C, T) and 'event_slices'.
# Output: .npz files containing 'DE' (N, C, 5) and 'labels' (N,).
# =============================================================================

import argparse
import math
import os
import warnings

import numpy as np
from scipy.signal import butter, lfilter
from tqdm import tqdm

# =============================================================================
# 1. Digital Signal Processing (DSP) Functions
# =============================================================================

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Generates coefficients for a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter to the input data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# =============================================================================
# 2. Differential Entropy (DE) Calculation
# =============================================================================

def compute_DE(signal):
    """
    Computes the Differential Entropy (DE) of a given signal segment.
    For a Gaussian distribution, DE is equivalent to the logarithm of the variance.
    
    Returns:
        variance (float): The variance of the signal.
        de (float): The differential entropy value.
    """
    # Ensure the signal length is sufficient to calculate variance
    if len(signal) <= 1:
        return 0.0, -np.inf 
        
    variance = np.var(signal, ddof=1)
    
    # Prevent math domain errors for zero or negative variance
    if variance <= 0:
        return variance, -np.inf
        
    de = math.log(2 * math.pi * math.e * variance) / 2
    return variance, de

# =============================================================================
# 3. Main Extraction Pipeline
# =============================================================================

def extract_de_from_1s_windows(data_root, save_root, num_subjects=18, frequency=128):
    """
    Reads pre-sliced EEG data, applies 5-band filtering, computes DE features, 
    and saves the extracted features to the target directory.
    
    Frequency Bands:
        - Delta: 0.1 - 4 Hz
        - Theta: 4 - 8 Hz
        - Alpha: 8 - 14 Hz
        - Beta:  14 - 31 Hz
        - Gamma: 31 - 50 Hz
    """
    os.makedirs(save_root, exist_ok=True)
    
    for participant in range(1, num_subjects + 1): 
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing Subject S{participant}...")

        # --- 1. Load Raw Sliced Data ---
        file_name = f'S{participant}_Dataset_1s.npz'
        file_path = os.path.join(data_root, file_name)
        
        if not os.path.exists(file_path):
            warnings.warn(f"File not found: {file_path}. Skipping Subject S{participant}.")
            continue
            
        try:
            data = np.load(file_path, allow_pickle=True)
            # Expected shape: (N_samples, 66, 128)
            EEG_data = data['eeg_slices'].astype(np.float64) 
            
            # Extract labels and shift to 0-indexed (e.g., 1,2 -> 0,1)
            labels = np.array([int(item[0]) for item in data['event_slices']]) - 1
            
        except Exception as e:
            warnings.warn(f"Error loading {file_path}: {e}. Skipping Subject S{participant}.")
            continue

        # --- 2. Initialize DE Storage Arrays ---
        num_samples = EEG_data.shape[0]
        num_channels = EEG_data.shape[1] # Typically 66 for DTU
        num_bands = 5
        
        decomposed_de = np.empty([num_samples, num_channels, num_bands])
        
        # Temporary buffers for the current sample
        de = np.empty([1, num_channels, num_bands])
        variances_temp = np.empty([1, num_channels, num_bands]) 

        print(f"Loaded {num_samples} samples, {num_channels} channels. Computing DE...")

        # --- 3. Iterate through each 1-second sample ---
        for sample in tqdm(range(num_samples), desc=f"Subject {participant}"):
            trial_signal = EEG_data[sample] # Shape: (Channels, Time)
            
            for channel in range(num_channels): 
                signal_1s = trial_signal[channel] # Shape: (Time,)
                
                # Apply bandpass filters
                delta_data = butter_bandpass_filter(signal_1s, 0.1, 4, frequency)
                theta_data = butter_bandpass_filter(signal_1s, 4, 8, frequency)
                alpha_data = butter_bandpass_filter(signal_1s, 8, 14, frequency)
                beta_data = butter_bandpass_filter(signal_1s, 14, 31, frequency)
                gamma_data = butter_bandpass_filter(signal_1s, 31, 50, frequency)

                # Compute and store DE features
                variances_temp[0, channel, 0], de[0, channel, 0] = compute_DE(delta_data)
                variances_temp[0, channel, 1], de[0, channel, 1] = compute_DE(theta_data)
                variances_temp[0, channel, 2], de[0, channel, 2] = compute_DE(alpha_data)
                variances_temp[0, channel, 3], de[0, channel, 3] = compute_DE(beta_data)
                variances_temp[0, channel, 4], de[0, channel, 4] = compute_DE(gamma_data)
            
            # Assign computed features to the main array
            decomposed_de[sample] = de

        # --- 4. Save Extracted Features ---
        save_file_path = os.path.join(save_root, f'S{participant}_DE_Features_1s.npz')
        np.savez(save_file_path, DE=decomposed_de, labels=labels)
        print(f"Successfully saved to: {save_file_path}")

    print("\n" + "="*60)
    print("DE feature extraction completed for all subjects.")
    print("="*60)

# =============================================================================
# 4. Execution Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Differential Entropy (DE) features from EEG signals.")
    
    # Configurable paths via command line
    parser.add_argument(
        '--data_root', 
        type=str, 
        default='./data/DTU', 
        help='Directory containing the raw 1s sliced .npz files.'
    )
    parser.add_argument(
        '--save_root', 
        type=str, 
        default='./data/DE_Features', 
        help='Target directory to save the extracted DE features.'
    )
    parser.add_argument(
        '--num_subjects', 
        type=int, 
        default=18, 
        help='Total number of subjects in the dataset.'
    )
    parser.add_argument(
        '--fs', 
        type=int, 
        default=128, 
        help='Sampling frequency of the EEG data.'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_root):
        print(f"!!! Error: The input directory '{args.data_root}' does not exist.")
        print("!!! Please specify the correct path using --data_root")
    else:
        extract_de_from_1s_windows(
            data_root=args.data_root, 
            save_root=args.save_root, 
            num_subjects=args.num_subjects,
            frequency=args.fs
        )