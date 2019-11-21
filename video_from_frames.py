import cv2
import os

input_video_name = "eating_resting_tiger_zoo"
labelled_frames_folder = os.path.join("D:/Sofi/Documents/ATRW_datatset/tiger_videos/",input_video_name + "_labelled/")
output_video_name = 'output.avi'
output_fps = 30

list_images = [img for img in os.listdir(labelled_frames_folder) if img.startswith("f") and img.endswith(".png")]

# get h,w,layers from 1t frame of first image
frame = cv2.imread(os.path.join(labelled_frames_folder, list_images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(os.path.join(labelled_frames_folder,output_video_name), 0,output_fps, (width,height))

for image in list_images:
    video.write(cv2.imread(os.path.join(labelled_frames_folder, image)))

cv2.destroyAllWindows()
video.release()