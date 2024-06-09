import os
import subprocess
import sys

# path = "/DATA_EDS2/chenjt2305/datasets/0409/group1"
# # os.system("cd /DATA_EDS2/chenjt2305/datasets/0320/group1")
# os.system("cd " + path)
# # path = "/DATA_EDS2/chenjt2305/datasets/0320/group1"
# for i in range(1, 8):
#     if not os.path.exists(path + "/new" + str(i)):
#         os.mkdir(path + "/new" + str(i))

#     subprocess.run(
#         f"ffmpeg -i {path}/group1video{i}.MP4 -r 1 -q:v 2 -f image2 {path}/new{i}/frame_%05d.png",
#         shell=True,
#     )
#     # -r for frame rate






import os

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
