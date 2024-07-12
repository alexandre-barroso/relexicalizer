# By Alexandre Barroso, for LL191 class by prof. dr. Plinio Barbosa @ Unicamp, 2024.
import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, activations
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.layers import LeakyReLU

# Enable XLA
tf.config.optimizer.set_jit(True)

### Variables

# Configuration Parameters
tfrecord_directory = '/content/dataset'
output_directory = '/content/drive/MyDrive/predictions'
model_checkpoint_directory = '/content/drive/MyDrive/model_checkpoints'

# Epochs
epochs = 5000
save_predictions_every_n_epochs = 1

# Shapes/dimensions
stft_magnitude_shape = (1025, 979)
stft_phase_shape = (1025, 979)

# Batch size
train_batch_size = 256  # Initial batch size
val_batch_size = 256
increase_step = 32
max_train_batch_size = 512

# Dataset size
n_dataset_train = 8000
n_dataset_val = 1000
n_dataset_test = 1000

# Sample size
total_samples = n_dataset_train + n_dataset_val + n_dataset_test
validation_samples = n_dataset_val

# Steps_per_epoch is at least 1
steps_per_epoch = max(1, total_samples // train_batch_size)
validation_steps = max(1, validation_samples // val_batch_size)

### Callbacks

# Saving predictions
class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, output_dir, save_freq):
        self.dataset = dataset
        self.output_dir = output_dir
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            for inputs, outputs in self.dataset.take(1):
                predictions = self.model.predict(inputs)
                if isinstance(predictions, list) and len(predictions) == 2:
                    predictions = (predictions[0], predictions[1])
                elif isinstance(predictions, np.ndarray):
                    predictions = (predictions, predictions)
                else:
                    raise ValueError("Unexpected prediction format.")

                index = np.random.randint(predictions[0].shape[0])  # Random example for saving
                prediction = (predictions[0][index], predictions[1][index])
                # Extract metadata for saving
                meta = {key: outputs[key][index] for key in outputs}
                output_path = os.path.join(self.output_dir, f"{meta['identifier'].numpy().decode('utf-8')}_epoch{epoch + 1}.npy")

                save_prediction(prediction, meta, output_path)

# Cyclic batch size
class CyclicBatchSize(Callback):
    def __init__(self, base_train_batch_size, base_val_batch_size, max_batch_size, step_size):
        super(CyclicBatchSize, self).__init__()
        self.base_train_batch_size = base_train_batch_size
        self.base_val_batch_size = base_val_batch_size
        self.max_batch_size = max_batch_size
        self.step_size = step_size
        self.train_batch_size = base_train_batch_size
        self.val_batch_size = base_val_batch_size

    def on_epoch_begin(self, epoch, logs=None):
        # Set the batch sizes at the beginning of each epoch
        self.set_batch_sizes(self.train_batch_size, self.val_batch_size)

        # Update batch sizes every 250 epochs
        if (epoch + 1) % 250 == 0:  # Check if the next epoch is a multiple of 250
            self.update_batch_sizes()

    def update_batch_sizes(self):
        # Increment training batch size
        new_train_batch_size = self.train_batch_size + self.step_size
        if new_train_batch_size <= self.max_batch_size:
            self.train_batch_size = new_train_batch_size
        else:
            self.train_batch_size = self.max_batch_size

        # Increment validation batch size
        new_val_batch_size = self.val_batch_size + self.step_size
        if new_val_batch_size <= self.max_batch_size:
            self.val_batch_size = new_val_batch_size
        else:
            self.val_batch_size = self.max_batch_size

    def set_batch_sizes(self, train_batch_size, val_batch_size):
        print(f"Train batch size {train_batch_size}")
        print(f"Val batch size: {val_batch_size}")
        if hasattr(self.model, 'train_dataset') and hasattr(self.model, 'val_dataset'):
            self.model.train_dataset = self.model.train_dataset.unbatch().batch(train_batch_size)
            self.model.val_dataset = self.model.val_dataset.unbatch().batch(val_batch_size)

# Cyclic learning rate
class CyclicLRWithRestarts(tf.keras.callbacks.Callback):
    def __init__(self, base_lr=0.0005, max_lr=0.00001, step_size=500, mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle', verbose=1):
        super(CyclicLRWithRestarts, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.verbose = verbose
        self.epoch_iterations = 0
        self.history = {}

        if self.scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

    def clr(self):
        cycle = np.floor(1 + self.epoch_iterations / (2 * self.step_size))
        x = np.abs(self.epoch_iterations / self.step_size - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.clr()
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.history.setdefault('lr', []).append(lr)
        self.epoch_iterations += 1

        if self.verbose > 0:
            print(f"\nEpoch {epoch+1}: setting learning rate to {lr:.5f}")

# Clean memory after every epoch
class MemoryCleanupCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()
        print(f"Memory cleanup at end of epoch {epoch+1}")

### Enable TPU Strategy
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print('Running on TPU')
except Exception as e:
    print('Failed to initialize TPU:', e)
    strategy = tf.distribute.MirroredStrategy()
    print('Running on GPU or CPU')

### Neural net architecture

def build_branch(input_shape, name, l2_reg):
    input_layer = layers.Input(shape=input_shape, name=name)
    x = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2_reg)(input_layer)
    x = layers.Bidirectional(layers.GRU(256, return_sequences=True, kernel_regularizer=l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    return input_layer, x


def build_model():
    print("Building model...")
    l2_reg = regularizers.l2(0.01)
    delex_stft_magnitude_input, x_stft_magnitude = build_branch(stft_magnitude_shape, 'delex_stft_magnitude_input', l2_reg)
    delex_stft_phase_input, x_stft_phase = build_branch(stft_phase_shape, 'delex_stft_phase_input', l2_reg)
    concatenated_stft = layers.Concatenate()([x_stft_magnitude, x_stft_phase])
    x_stft = layers.GRU(512, return_sequences=True, kernel_regularizer=l2_reg)(concatenated_stft)
    x_stft = layers.BatchNormalization()(x_stft)
    x_stft = layers.Dropout(0.5)(x_stft)
    stft_magnitude_output = layers.TimeDistributed(layers.Dense(979, activation='linear'), name='original_stft_magnitude')(x_stft)
    stft_phase_output = layers.TimeDistributed(layers.Dense(979, activation='linear'), name='original_stft_phase')(x_stft)
    model = models.Model(inputs=[delex_stft_magnitude_input, delex_stft_phase_input], outputs=[stft_magnitude_output, stft_phase_output])
    print("Model built.")
    return model

### Reading and preparing dataste

# Parses a single example from a TFRecord
def parse_tf_example(proto):
    feature_description = {
        'sr': tf.io.FixedLenFeature([], tf.int64),
        'delex_stft_magnitude': tf.io.FixedLenFeature([], tf.string),
        'delex_stft_phase': tf.io.FixedLenFeature([], tf.string),
        'original_stft_magnitude': tf.io.FixedLenFeature([], tf.string),
        'original_stft_phase': tf.io.FixedLenFeature([], tf.string),
        'identifier': tf.io.FixedLenFeature([], tf.string),
        'original_stft_magnitude_original_length': tf.io.FixedLenFeature([], tf.int64),
        'original_stft_phase_original_length': tf.io.FixedLenFeature([], tf.int64)
    }

    example = tf.io.parse_single_example(proto, feature_description)
    delex_stft_magnitude = tf.io.parse_tensor(example['delex_stft_magnitude'], out_type=tf.float32)
    delex_stft_phase = tf.io.parse_tensor(example['delex_stft_phase'], out_type=tf.float32)
    original_stft_magnitude = tf.io.parse_tensor(example['original_stft_magnitude'], out_type=tf.float32)
    original_stft_phase = tf.io.parse_tensor(example['original_stft_phase'], out_type=tf.float32)

    delex_stft_magnitude = tf.ensure_shape(delex_stft_magnitude, stft_magnitude_shape)
    delex_stft_phase = tf.ensure_shape(delex_stft_phase, stft_phase_shape)
    original_stft_magnitude = tf.ensure_shape(original_stft_magnitude, stft_magnitude_shape)
    original_stft_phase = tf.ensure_shape(original_stft_phase, stft_phase_shape)

    inputs = {'delex_stft_magnitude_input': delex_stft_magnitude, 'delex_stft_phase_input': delex_stft_phase}

    outputs = {
        'original_stft_magnitude': original_stft_magnitude,
        'original_stft_phase': original_stft_phase,
        'sr': example['sr'],
        'identifier': example['identifier'],
        'original_stft_magnitude_length': example['original_stft_magnitude_original_length'],
        'original_stft_phase_length': example['original_stft_phase_original_length']
    }
    return inputs, outputs

def create_dataset(tfrecord_files, batch_size, include_metadata=False):
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tf_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if include_metadata:
        dataset = dataset.map(lambda inputs, outputs: (inputs, outputs))
    else:
        dataset = dataset.map(lambda inputs, outputs: (
            inputs,
            {
                'original_stft_magnitude': outputs['original_stft_magnitude'],
                'original_stft_phase': outputs['original_stft_phase']
            }
        ))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def save_prediction(prediction, metadata, output_path):
        original_stft_magnitude, original_stft_phase = prediction
        if isinstance(original_stft_magnitude, tf.Tensor):
            original_stft_magnitude = original_stft_magnitude.numpy()
        if isinstance(original_stft_phase, tf.Tensor):
            original_stft_phase = original_stft_phase.numpy()

        features = {
            'stft_magnitude': original_stft_magnitude,
            'stft_phase': original_stft_phase,
            'sr': metadata['sr'].numpy() if isinstance(metadata['sr'], tf.Tensor) else metadata['sr'],
            'identifier': metadata['identifier'].numpy().decode('utf-8') if isinstance(metadata['identifier'], tf.Tensor) else metadata['identifier'],
            'original_stft_magnitude_length': metadata['original_stft_magnitude_length'].numpy() if isinstance(metadata['original_stft_magnitude_length'], tf.Tensor) else metadata['original_stft_magnitude_length'],
            'original_stft_phase_length': metadata['original_stft_phase_length'].numpy() if isinstance(metadata['original_stft_phase_length'], tf.Tensor) else metadata['original_stft_phase_length']
        }
        np.save(output_path, features)
        print(f"Saved prediction to {output_path}")

### Setup environment and paths

os.makedirs(output_directory, exist_ok=True)
os.makedirs(model_checkpoint_directory, exist_ok=True)
train_tfrecord_directory = os.path.join(tfrecord_directory, 'train')
val_tfrecord_directory = os.path.join(tfrecord_directory, 'val')
test_tfrecord_directory = os.path.join(tfrecord_directory, 'test')
train_tfrecord_files = [os.path.join(train_tfrecord_directory, file) for file in os.listdir(train_tfrecord_directory) if file.endswith('.tfrecord')]
val_tfrecord_files = [os.path.join(val_tfrecord_directory, file) for file in os.listdir(val_tfrecord_directory) if file.endswith('.tfrecord')]
test_tfrecord_files = [os.path.join(test_tfrecord_directory, file) for file in os.listdir(test_tfrecord_directory) if file.endswith('.tfrecord')]

# Use only the first 'n' files
train_tfrecord_files = train_tfrecord_files[:n_dataset_train]
val_tfrecord_files = val_tfrecord_files[:n_dataset_val]
test_tfrecord_files = test_tfrecord_files[:n_dataset_test]

# Define and create datasets with initial batch sizes
print("Creating train_dataset dataset...")
train_dataset = create_dataset(train_tfrecord_files, train_batch_size)

print("Creating val_dataset_no_metadata dataset...")
val_dataset_no_metadata = create_dataset(val_tfrecord_files, val_batch_size)

print("Creating val_dataset_with_metadata dataset...")
val_dataset_with_metadata = create_dataset(val_tfrecord_files, val_batch_size, include_metadata=True)

print('Dataset(s) created.')

### Loss function

def waveform_loss(y_true, y_pred):
    def to_waveform(magnitude, phase):
        stft = tf.complex(magnitude * tf.cos(phase), magnitude * tf.sin(phase))
        return tf.signal.inverse_stft(stft, frame_length=2048, frame_step=512)

    true_magnitude = y_true[0]
    true_phase = y_true[1]
    pred_magnitude = y_pred[0]
    pred_phase = y_pred[1]

    true_waveform = to_waveform(true_magnitude, true_phase)
    pred_waveform = to_waveform(pred_magnitude, pred_phase)

    # Use L1 loss for less sensitivity to outliers and stable training
    loss = tf.reduce_mean(tf.abs(true_waveform - pred_waveform))
    return loss

def spectral_convergence_loss(y_true, y_pred):
    def compute_magnitude(magnitude, phase):
        stft = tf.complex(magnitude * tf.cos(phase), magnitude * tf.sin(phase))
        return tf.abs(stft)

    true_magnitude = y_true[0]
    true_phase = y_true[1]
    pred_magnitude = y_pred[0]
    pred_phase = y_pred[1]

    magnitude_true = compute_magnitude(true_magnitude, true_phase)
    magnitude_pred = compute_magnitude(pred_magnitude, pred_phase)

    # Improve numerical stability by adding a small constant inside the logarithm
    loss = tf.reduce_mean(tf.abs(tf.math.log(magnitude_true + 1e-10) - tf.math.log(magnitude_pred + 1e-10)))
    return loss

def phase_loss(y_true, y_pred):
    true_phase = y_true[1]
    pred_phase = y_pred[1]
    # Use cosine distance to ensure smooth and continuous loss for phases
    loss = tf.reduce_mean(1 - tf.cos(true_phase - pred_phase))
    return loss

def perceptual_loss(y_true, y_pred):
    def to_mel_spectrogram(magnitude, phase):
        stft = tf.complex(magnitude * tf.cos(phase), magnitude * tf.sin(phase))
        waveform = tf.signal.inverse_stft(stft, frame_length=2048, frame_step=512)
        spectrogram = tf.abs(tf.signal.stft(waveform, frame_length=2048, frame_step=512))
        mel_weights = tf.signal.linear_to_mel_weight_matrix(128, 1025, 16000, 20, 4000)
        mel_spectrogram = tf.tensordot(spectrogram, mel_weights, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(mel_weights.shape[-1:]))
        return tf.math.log(mel_spectrogram + 1e-6)

    true_magnitude = y_true[0]
    true_phase = y_true[1]
    pred_magnitude = y_pred[0]
    pred_phase = y_pred[1]

    true_mel = to_mel_spectrogram(true_magnitude, true_phase)
    pred_mel = to_mel_spectrogram(pred_magnitude, pred_phase)

    loss = tf.reduce_mean(tf.square(true_mel - pred_mel))
    return loss

def combined_loss(y_true, y_pred):
    loss_waveform = waveform_loss(y_true, y_pred)
    loss_spectral = spectral_convergence_loss(y_true, y_pred)
    loss_phase = phase_loss(y_true, y_pred)
    loss_perceptual = perceptual_loss(y_true, y_pred)

    return 0.1 * loss_waveform + 0.3 * loss_spectral + 0.5 * loss_perceptual + 0.1 * loss_phase

### Initialize the model and callbacks

# Define model within the TPU strategy scope
with strategy.scope():
    model = build_model()
    base_learning_rate = 0.01
    learning_rate = base_learning_rate * (train_batch_size / 256)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, clipvalue=1.0, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=combined_loss)

# Load checkpoint if it exists
checkpoint_path = os.path.join(model_checkpoint_directory, "model_checkpoint.h5")
if os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model.load_weights(checkpoint_path)
else:
    print("No checkpoint found. Starting from scratch.")

model.train_dataset = train_dataset
model.val_dataset = val_dataset_no_metadata

# Initialize callbacks
cyclic_lr_with_restarts = CyclicLRWithRestarts(base_lr=0.01, max_lr=0.00001, step_size=100, mode='triangular')

early_stopping = EarlyStopping(monitor='val_loss', patience=2000, restore_best_weights=True)

checkpoint_callback = ModelCheckpoint(filepath=os.path.join(model_checkpoint_directory, 'model_checkpoint.h5'), monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)

prediction_callback = PredictionCallback(val_dataset_with_metadata, output_directory, save_predictions_every_n_epochs)

cyclic_batch_size_callback = CyclicBatchSize(base_train_batch_size=train_batch_size, base_val_batch_size=val_batch_size, max_batch_size=max_train_batch_size, step_size=increase_step)

memory_cleanup_callback = MemoryCleanupCallback()

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=500, min_lr=1e-6, verbose=1)

# Clean memory before training
def check_and_clean_memory():
    K.clear_session()
    gc.collect()

check_and_clean_memory()

# Start training
model.fit(
    model.train_dataset,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=model.val_dataset,
    validation_steps=validation_steps,
    callbacks=[
        checkpoint_callback,
        early_stopping,
        prediction_callback,
        cyclic_lr_with_restarts,
        cyclic_batch_size_callback,
        memory_cleanup_callback, # Beware of cleaning between (in between?) epochs and losing information
        lr_scheduler
    ]
)

print('Training finished.')
