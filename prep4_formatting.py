import os
import numpy as np
import librosa

def pad_or_truncate(array, target_length):
    original_length = array.shape[1]
    if original_length < target_length:
        padding = target_length - original_length
        padded_array = np.pad(array, ((0, 0), (0, padding)), mode='constant')
        return padded_array, original_length, padding
    else:
        truncated_array = array[:, :target_length]
        return truncated_array, original_length, 0

def process_and_save_features(input_dir, output_dir, mfcc_length=380, stft_length=380):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                try:
                    features = np.load(file_path, allow_pickle=True).item()

                    mfccs, mfcc_original_length, mfcc_padding = pad_or_truncate(features['mfccs'], mfcc_length)
                    stft, stft_original_length, stft_padding = pad_or_truncate(features['stft'], stft_length)

                    processed_features = {
                        'mfccs': mfccs,
                        'stft': stft,
                        'sr': features['sr'],
                        'downsampling_factor': features['downsampling_factor'],
                        'mfcc_original_length': mfcc_original_length,
                        'mfcc_padding': mfcc_padding,
                        'stft_original_length': stft_original_length,
                        'stft_padding': stft_padding
                    }


                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    np.save(output_path, processed_features)
                    print(f"Processed and saved {output_path}")

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

input_directory = 'data_features'
output_directory = 'data_processed'
process_and_save_features(input_directory, output_directory)

