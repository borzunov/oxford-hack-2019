from array import array
import os
from PIL import Image, ImageDraw, ImageFont
import sys
import time
import cv2

# Fetch video names
videos_folder = "D:/Sofi/Documents/ATRW_datatset/tiger_videos/"
videos_list = os.listdir(videos_folder)

for video_name in videos_list:
    print("Processing video: {}".format(video_name))

    # get video
    video_path = os.path.join(videos_folder, video_name)
    vidcap = cv2.VideoCapture(video_path)

    # print frame rate
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frame rate: {}".format(fps)) # 30 fps
    print("Total frames: {}".format(length))

    # make subfolder
    frames_folder = os.path.join(video_path.split(".")[0] + "/")  # ,"/frame%04d.jpg" % count, image))
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    # Read frame
    count = 0
    success, image = vidcap.read()
    while success:

        # save frame as JPEG file
        frame_path = os.path.join(frames_folder + ("frame%04d.jpg" % count))
        cv2.imwrite(frame_path, image)

        # read next frame
        success, image = vidcap.read()

        #print('Read a new frame: ', success)
        count += 1

