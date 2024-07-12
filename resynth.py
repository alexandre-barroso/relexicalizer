import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Set the gain for amplifying the audio. Increase or decrease as needed.
gain = 5.0

def load_prediction(file_path):
    try:
        data = np.load(file_path, allow_pickle=True).item()
        stft_magnitude = data['stft_magnitude']
        stft_phase = data['stft_phase']
        sr = data['sr']
        identifier = data['identifier']
        original_stft_magnitude_length = data['original_stft_magnitude_length']
        original_stft_phase_length = data['original_stft_phase_length']
        if stft_magnitude is not None and stft_phase is not None and sr is not None:
            return stft_magnitude, stft_phase, sr, identifier, original_stft_magnitude_length, original_stft_phase_length
        else:
            return None, None, None, None, None, None
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None, None, None, None, None, None

def denormalize(data, mean, std):
    return (data * std) + mean

def check_sample_rate(sr, min_freq=80, max_freq=11000):
    if sr < 2 * max_freq:
        print(f"Sample rate {sr} Hz is less than twice the max frequency of human voice. Adjusting...")
        return 22050
    return sr

def reconstruct_audio(stft_magnitude, stft_phase, original_magnitude_length, original_phase_length, mean, std, gain, sr):
    if stft_magnitude is None or stft_phase is None:
        print("Invalid STFT data provided.")
        return None

    # Check if sample rate is sufficient for human voice frequency
    sr = check_sample_rate(sr)

    # Remove the padding using the original lengths
    stft_magnitude = stft_magnitude[:, :original_magnitude_length]
    stft_phase = stft_phase[:, :original_phase_length]

    # Denormalize the STFT magnitude
    stft_magnitude = denormalize(stft_magnitude, mean, std)

    # Reconstruct the complex STFT matrix
    stft = stft_magnitude * np.exp(1j * stft_phase)

    # Perform the inverse STFT to get the time-domain signal
    try:
        y = librosa.istft(stft, hop_length=512, win_length=1024)  # Ensure these parameters match the original STFT extraction
    except Exception as e:
        print(f"Failed to perform iSTFT: {e}")
        return None

    # Handle case where reconstruction fails
    if y is None or np.max(np.abs(y)) == 0:
        print("Invalid audio reconstruction.")
        return None

    # Normalize the audio amplitude to be within the range [-1, 1]
    y /= np.max(np.abs(y))

    # Apply the specified gain
    y *= gain

    # Log the reconstructed audio stats
    print(f"Reconstructed audio shape: {y.shape}")
    print(f"Reconstructed audio mean: {np.mean(y)}")
    print(f"Reconstructed audio std: {np.std(y)}")
    print(f"Reconstructed audio max amplitude: {np.max(y)}")
    print(f"Reconstructed audio min amplitude: {np.min(y)}")

    return y

def save_audio(y, sr, output_path):
    if y is not None:
        sf.write(output_path, y, sr)
        print(f'Saved reconstructed audio to {output_path}')
    else:
        print(f"Skipping save for {output_path}, invalid audio.")

# Directory containing the saved predictions
predictions_directory = 'predictions'
output_directory = 'reconstructed_wavs'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Mean and std used during normalization when feature extracting
mean = 0.2472354034009002
std = 3.226139918522755

# Process each prediction file
for root, _, files in os.walk(predictions_directory):
    for file in tqdm(files):
        if file.endswith('.npy'):
            file_path = os.path.join(root, file)
            result = load_prediction(file_path)
            if result:
                stft_magnitude, stft_phase, sr, identifier, original_magnitude_length, original_phase_length = result
                y = reconstruct_audio(stft_magnitude, stft_phase, original_magnitude_length, original_phase_length, mean, std, gain, sr)
                if y is not None:
                    relative_path = os.path.relpath(root, predictions_directory)
                    output_subdir = os.path.join(output_directory, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_path = os.path.join(output_subdir, file.replace('.npy', '.wav'))
                    save_audio(y, sr, output_path)
