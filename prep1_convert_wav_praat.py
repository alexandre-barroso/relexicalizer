import os
import subprocess
from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd

def convert_and_delexicalize_mp3(input_folder, output_folder, praat_script_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file in tqdm(os.listdir(input_folder), desc="Converting and delexicalizing MP3 to WAV"):
        if file.endswith('.mp3'):
            mp3_path = os.path.join(input_folder, file)
            temporary_wav_path = os.path.join(output_folder, "temp_" + file.replace('.mp3', '.wav'))
            delexicalized_wav_path = os.path.join(output_folder, file.replace('.mp3', '_purr.wav'))
            try:
                sound = AudioSegment.from_mp3(mp3_path)
                sound.export(temporary_wav_path, format='wav')
                delexicalize_audio(temporary_wav_path, delexicalized_wav_path, praat_script_path)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {mp3_path}: {e}")
                continue
            finally:
                if os.path.exists(temporary_wav_path):
                    os.remove(temporary_wav_path)

def delexicalize_audio(wav_path, output_path, praat_script_path):
    command = ['praat', '--run', praat_script_path, wav_path, output_path]
    subprocess.run(command, check=True)

def prepare_transcriptions(df, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preparing transcriptions"):
        file_name = row['path'].replace('.mp3', '_purr.txt')
        with open(os.path.join(output_folder, file_name), 'w') as f:
            f.write(row['sentence'])

if __name__ == '__main__':
    input_folder = 'data/clips'
    output_folder = 'data/wav_clips'
    praat_script_path = 'delexicalize.praat'
    convert_and_delexicalize_mp3(input_folder, output_folder, praat_script_path)
    train_df = pd.read_csv('data/metadata/train.tsv', sep='\t')
    dev_df = pd.read_csv('data/metadata/dev.tsv', sep='\t')
    test_df = pd.read_csv('data/metadata/test.tsv', sep='\t')
    validated_df = pd.read_csv('data/metadata/validated.tsv', sep='\t')
    prepare_transcriptions(train_df, output_folder)
    prepare_transcriptions(dev_df, output_folder)
    prepare_transcriptions(test_df, output_folder)
    prepare_transcriptions(validated_df, output_folder)
