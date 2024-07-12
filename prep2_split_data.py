import os
import shutil
from sklearn.model_selection import train_test_split

def ensure_1_to_1_to_1_ratio(input_dir):
    base_names = set()
    for f in os.listdir(input_dir):
        if f.startswith("common_voice_pt_") and not f.endswith('_purr.wav') and not f.endswith('_purr.txt'):
            unique_number = f[len("common_voice_pt_"):-len(".wav")]
            base_name = f"common_voice_pt_{unique_number}"
            if all(os.path.exists(os.path.join(input_dir, base_name + suffix)) for suffix in ['.wav', '_purr.wav', '_purr.txt']):
                base_names.add(base_name)
            else:
                for suffix in ['.wav', '_purr.wav', '_purr.txt']:
                    file_path = os.path.join(input_dir, base_name + suffix)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                print(f"Removed incomplete group for base name: {base_name}")

    return list(base_names)

def split_data(input_dir, output_dirs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    base_names = ensure_1_to_1_to_1_ratio(input_dir)
    for dir in output_dirs.values():
        for subset in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dir, subset), exist_ok=True)

    print(f"Total number of base filenames: {len(base_names)}")

    if len(base_names) < 3:
        print("Not enough files to split into train, val, and test sets.")
        return
    train_files, temp_files = train_test_split(base_names, test_size=1 - train_ratio, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)
    print(f"Number of train files: {len(train_files)}")
    print(f"Number of val files: {len(val_files)}")
    print(f"Number of test files: {len(test_files)}")
    subsets = {'train': train_files, 'val': val_files, 'test': test_files}
    for subset, files in subsets.items():
        for base_name in files:
            for suffix in ['.wav', '_purr.wav', '_purr.txt']:
                source_path = os.path.join(input_dir, base_name + suffix)
                dest_path = os.path.join(output_dirs[suffix], subset, base_name + suffix)

                if not os.path.exists(dest_path):
                    if os.path.exists(source_path):
                        shutil.move(source_path, dest_path)
                    else:
                        print(f"Warning: Expected file {source_path} not found.")
                else:
                    print(f"File {dest_path} already exists, skipping.")

output_dirs = {
    '.wav': 'data/original_splits',
    '_purr.wav': 'data/delexicalized_splits',
    '_purr.txt': 'data/text_splits'
}

split_data('data/wav_clips', output_dirs)
