import os

def rename_files(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            old_path = os.path.join(subdir, file)
            new_filename = file
            if new_filename.startswith("common_voice_pt_"):
                new_filename = new_filename[len("common_voice_pt_"):]
            new_filename = new_filename.replace("_purr.phonetic.txt", "_phonetic.txt")
            new_path = os.path.join(subdir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} to {new_path}")

root_dir = 'data'
rename_files(root_dir)
