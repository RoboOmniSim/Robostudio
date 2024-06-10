import os
import subprocess
import sys

# Set the directory containing the images
directory = '/home/lou/Downloads/zero_pose/pro'

# Get a list of files in the directory
files = os.listdir(directory)
files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # Filter for image files only

# Sort the files if necessary, to ensure correct ordering
files.sort()

# Define the starting index
start_index = 313

# Loop through the files and rename them
for i, filename in enumerate(files):
    # Create the new filename
    new_filename = f'frame_{start_index + i:05d}.png'
    # Create the full paths
    old_file = os.path.join(directory, filename)
    new_file = os.path.join(directory, new_filename)
    
    # Rename the file
    os.rename(old_file, new_file)

    # Optional: Print the old and new file names to verify
    print(f'Renamed "{filename}" to "{new_filename}"')
