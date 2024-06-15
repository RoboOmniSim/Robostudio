import cv2
import os

def create_video_from_images(image_folder, output_video, fps):
    # Get a list of all images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # Sort images if necessary

    # Read the first image to get the size
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        video.write(img)

    # Release the video writer object
    video.release()


if __name__ == '__main__':

    # # Example usage of novel_pose
    # image_folder = '/home/lou/gs/nerfstudio/renders/novel_pose_dynamic/frame_00309.png/rgb'
    # output_video = '/home/lou/gs/nerfstudio/renders/novel_pose_dynamic/frame_00309.mp4'


    # Example usage of push_box
    # image_folder = '/home/lou/gs/nerfstudio/renders/push_box_dynamic/frame_00293.png/rgb'
    # output_video = '/home/lou/gs/nerfstudio/renders/push_box_dynamic/frame_00293.mp4'


    image_folder = '/home/lou/gs/Robostudio/dataset/issac2sim/render/frame_00283.png/rgb'
    output_video = '/home/lou/gs/Robostudio/dataset/issac2sim/render/frame_00283.mp4'

    # if os.path.exists(output_video) is False:
    #     os.makedirs(output_video)
    fps = 10  # Frames per second

    create_video_from_images(image_folder, output_video, fps)