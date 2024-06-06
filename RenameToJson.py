import os

folder_path = 'D:/New folder/Extracted'

# List all files in the folder
files = os.listdir(folder_path)

# Loop through each file
for file in files:
    # Split the file name and extension
    name, ext = os.path.splitext(file)
    
    # Check if the file name already ends with .json
    if ext != '.json':
        # Rename the file by adding .json at the end
        new_name = name + '.json'
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))

print("All files renamed successfully!")