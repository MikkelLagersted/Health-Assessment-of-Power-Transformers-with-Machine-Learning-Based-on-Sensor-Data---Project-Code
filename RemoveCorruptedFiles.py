import os
import shutil
import json

def check_json_files(directory, target_directory):
    # Ensure the target directory exists, if not, create it
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    unimportable_files = []

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    
             
            except Exception as e:
                print(f"Failed to import {filename}: {e}")
                unimportable_files.append(filename)
                # Move the problematic file to the target directory
                shutil.move(file_path, os.path.join(target_directory, filename))
                print("{filename} moved")

    return unimportable_files

# Specify the directory to check where the JSON files are stored
folder_path = 'D:/New folder/Extracted'
# Specify the directory where unimportable files should be moved
target_folder_path = 'D:/Ishoj/Corrupt'
unimportable = check_json_files(folder_path, target_folder_path)
print("Unimportable files:", unimportable)