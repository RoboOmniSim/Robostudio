import subprocess

def extract_first_frame(video_path, output_path):
    command = [
        'ffmpeg',
        '-i', video_path,   # Input video path
        '-frames:v', '1',   # Extract one frame
        '-ss', '50',      # Seek to the 10.5 second
        '-q:v', '2',        # Set the quality of the output image
        output_path         # Output image path
    ]
    subprocess.run(command, check=True)

# Example usage
    

if __name__ == '__main__':
    video_path = ''
    output_path = ''
    extract_first_frame(video_path, output_path)
