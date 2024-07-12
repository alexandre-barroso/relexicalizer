# README

## Overview
This script, developed for the LL191 class by Prof. Dr. Plinio Barbosa at Unicamp in 2024, trains a neural network model for audio processing using TensorFlow. The script includes custom callbacks for saving predictions, adjusting batch sizes and learning rates cyclically, and performing memory cleanup after each epoch.

## Main Requirements
- Python 3.6+
- TensorFlow 2.x
- NumPy

## Configuration
### Directories
- **tfrecord_directory**: Path to the directory containing the TFRecord files (`/content/dataset`).
- **output_directory**: Path to the directory where predictions will be saved (`/content/drive/MyDrive/predictions`).
- **model_checkpoint_directory**: Path to the directory where model checkpoints will be saved (`/content/drive/MyDrive/model_checkpoints`).

### Training Parameters
- **epochs**: 5000
- **save_predictions_every_n_epochs**: Save predictions every epoch.
- **train_batch_size**: Initial batch size for training (256).
- **val_batch_size**: Initial batch size for validation (256).
- **increase_step**: Increment step for batch size (32).
- **max_train_batch_size**: Maximum batch size for training (512).

### Dataset Sizes
- **n_dataset_train**: Number of training samples (240).
- **n_dataset_val**: Number of validation samples (30).
- **n_dataset_test**: Number of test samples (30).
- **total_samples**: Total number of samples (300).
- **validation_samples**: Number of validation samples (30).

### Shapes
- **stft_magnitude_shape**: Shape of the STFT magnitude input (1025, 979).
- **stft_phase_shape**: Shape of the STFT phase input (1025, 979).

### Steps
- **steps_per_epoch**: Number of steps per epoch (total_samples // train_batch_size).
- **validation_steps**: Number of validation steps (validation_samples // val_batch_size).

## Callbacks
### PredictionCallback
This callback saves model predictions to the specified directory at the end of every `save_predictions_every_n_epochs` epochs.

### CyclicBatchSize
This callback adjusts the batch size cyclically every 250 epochs, incrementing the batch size by `increase_step` until it reaches `max_train_batch_size`.

### CyclicLRWithRestarts
This callback cyclically adjusts the learning rate based on a triangular learning rate schedule.

### MemoryCleanupCallback
This callback performs memory cleanup at the end of each epoch to manage memory usage efficiently.

## TPU/GPU Strategy
The script attempts to initialize a TPU strategy. If TPU initialization fails, it falls back to a mirrored strategy using GPU or CPU.

## Neural Network Architecture
### build_branch
Defines a branch of the neural network with:
- Convolutional layer with 256 filters and ReLU activation.
- Bidirectional GRU layer with 256 units.
- Batch normalization and dropout layers.

### build_model
Combines two branches (for STFT magnitude and phase) and concatenates their outputs. The combined output is passed through additional GRU, batch normalization, and dropout layers. The model outputs time-distributed dense layers for the STFT magnitude and phase.

## Dataset Preparation
### parse_tf_example
Parses a single example from a TFRecord file, ensuring the tensors are shaped correctly.

### create_dataset
Creates a TensorFlow dataset from TFRecord files, with options to include or exclude metadata. The dataset is shuffled, batched, repeated, and prefetched.

## Saving Predictions
### save_prediction
Saves model predictions along with metadata to the specified output path.

## Environment Setup
- Creates necessary directories for output and model checkpoints.
- Defines paths for training, validation, and test TFRecord files.
- Creates datasets with initial batch sizes for training and validation.

## Loss Functions
### waveform_loss
Calculates the L1 loss between the true and predicted waveforms by converting STFT magnitudes and phases back to the time-domain waveforms.

### spectral_convergence_loss
Calculates the logarithmic difference between the true and predicted STFT magnitudes.

### phase_loss
Calculates the cosine distance between the true and predicted phases.

### perceptual_loss
Calculates the mean squared error between the true and predicted Mel-spectrograms, derived from the waveforms.

### combined_loss
Combines the above loss functions with specific weights.

## Training
1. **Setup the Environment**:
   - Initializes the TPU or GPU/CPU strategy.
   - Creates directories for output and model checkpoints.
   - Defines paths for TFRecord files.
   - Creates training and validation datasets.
   - Defines custom callbacks.

2. **Build the Model**:
   - Defines and compiles the model within the TPU/GPU strategy scope.
   - Uses the AdamW optimizer with a base learning rate adjusted for the batch size.
   - Compiles the model with the combined loss function.

3. **Load Checkpoint**:
   - Loads model weights from a checkpoint if available.

4. **Initialize Callbacks**:
   - Cyclic learning rate with restarts.
   - Early stopping based on validation loss.
   - Model checkpoint saving the best model based on validation loss.
   - Prediction callback to save predictions.
   - Cyclic batch size adjustment.
   - Memory cleanup after each epoch.
   - Learning rate reduction on plateau.

5. **Clean Memory**:
   - Clears the Keras backend session and garbage collects before training.

6. **Train the Model**:
   - Fits the model on the training dataset with validation, using the defined callbacks.

## Example Usage
1. **Install Dependencies**:
    ```bash
    pip3 install tensorflow numpy
    ```

2. **Run the Script**:
    ```bash
    python3 neuralnet.py
    ```

3. **Monitor Training**:
    - Check the output directory for saved predictions.
    - Check the model checkpoint directory for saved model checkpoints.
  
## Resynth Script
This script reconstructs audio from the saved predictions.

### Script Overview
The `resynth` script loads predictions, reconstructs the audio, and saves it as `.wav` files.

### Requirements
- Python 3.6+
- NumPy
- Librosa
- SoundFile
- TQDM

### Configuration
- **predictions_directory**: Directory containing the saved predictions (`predictions`).
- **output_directory**: Directory to save the reconstructed audio files (`reconstructed_wavs`).
- **gain**: Gain factor for amplifying the audio (default is 5.0).
- **mean**: Mean used during normalization (0.2472354034009002).
- **std**: Standard deviation used during normalization (3.226139918522755).

### Functions
#### load_prediction
Loads a prediction file and extracts the STFT magnitude, phase, sample rate, identifier, and original lengths.

#### denormalize
Denormalizes the data using the provided mean and standard deviation.

#### check_sample_rate
Ensures the sample rate is sufficient for human voice frequency. Adjusts if necessary.

#### reconstruct_audio
Reconstructs the audio from STFT magnitude and phase:
- Checks sample rate.
- Removes padding.
- Denormalizes STFT magnitude.
- Reconstructs complex STFT matrix.
- Performs inverse STFT to obtain the time-domain signal.
- Normalizes and applies gain to the audio.

#### save_audio
Saves the reconstructed audio to the specified output path.

### Running the Script
1. **Ensure Dependencies**:
    ```bash
    pip install numpy librosa soundfile tqdm
    ```

2. **Run the Script**:
    ```bash
    python resynth.py
    ```

3. **Check the Output**:
    - The reconstructed audio files will be saved in the `reconstructed_wavs` directory.

## Author
Alexandre Barroso

## License
This project is licensed under the GNU License.

## Author
Alexandre Menezes Barroso (2024)

## License
This project is licensed under the GNU License. Feel free to change, add, remove, etc.
