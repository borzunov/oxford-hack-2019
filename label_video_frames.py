from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image, ImageDraw, ImageFont
import sys
import time
import cv2

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

################################################################################3
# Add your Computer Vision subscription key to your environment variables.
# In console: setx COMPUTER_VISION_SUBSCRIPTION_KEY key 
#             setx COMPUTER_VISION_ENDPOINT https://sm-cv.cognitiveservices.azure.com/
# https://docs.microsoft.com/es-es/azure/cognitive-services/cognitive-services-apis-create-account?tabs=multiservice%2Cwindows#configure-an-environment-variable-for-authentication

if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
    subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
else:
    print(
        "\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()
# Add your Computer Vision endpoint to your environment variables.
if 'COMPUTER_VISION_ENDPOINT' in os.environ:
    endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
else:
    print(
        "\nSet the COMPUTER_VISION_ENDPOINT environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()
##################################################################3


### Keras model
label_map = ['eating', 'resting', 'grooming', 'running', 'sleeping', 'walking', 'swimming']
#['eating', 'resting', 'grooming', 'running', 'sleeping', 'walking']
label_dict = {ky:0 for ky in label_map}
label_dict_norm = label_dict.copy()

# load model
model = load_model('model_v3.h5')



### Run obj detection
# Create a client
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Select folder with frames
video_filename = "eating_resting_tiger_zoo"
print("Video: {}".format(video_filename))
frames_folder = os.path.join("D:/Sofi/Documents/ATRW_datatset/tiger_videos/",video_filename) #"D:/Sofi/Documents/ATRW_datatset/trainval0" #
# output folder
labelled_frames_folder = frames_folder + "_labelled/"
if not os.path.exists(labelled_frames_folder):
    os.makedirs(labelled_frames_folder)

# loop thru frames in folder
list_local_imgs = os.listdir(frames_folder)
print("Total number of frames: {}".format(len(list_local_imgs)))
count = 0
step_frames = 30
ini_frame = 0
end_frame = len(list_local_imgs)
for i in range(ini_frame, end_frame, 1):  #local_img in list_local_imgs: #

    # # print frame
    local_img = list_local_imgs[i]
    print(list_local_imgs[i])

    # img
    #local_image = open(os.path.join(frames_folder,local_img),"rb")
    pth = os.path.join(frames_folder,local_img)
    img = Image.open(pth)

    # Classify only every 30 frames
    if i % step_frames == 0:
        # Call API with local image
        detect_objects_results_remote = computervision_client.detect_objects_in_stream(open(pth,'rb'))
        # pause
        #time.sleep(5.0)

        # # Get image as array
        # img = Image.open(local_image)

        # Print detected objects and bounding box coords
        if len(detect_objects_results_remote.objects) == 0:
            print("No objects detected.")

        else:
            for obj in detect_objects_results_remote.objects:
                # if detected object is tiger: feed crop to action classifier
                if obj.object_property in ["tiger", "animal", "mammal"]:
                    # Crop bbox with margin
                    margin = 0
                    img_crop = img.crop(((1.0-margin)*obj.rectangle.x,
                                         (1.0-margin)*obj.rectangle.y,
                                         (1.0+margin)*(obj.rectangle.x + obj.rectangle.w),
                                         (1.0+margin)*(obj.rectangle.y + obj.rectangle.h)))
                    #######################################################################
                    # Feed to action classifier

                    # scale, normalise, add batch dimension
                    img_crop = img_crop.resize((224, 224))
                    np_img_crop = np.array(img_crop) / 255
                    np_img_crop = np.expand_dims(np_img_crop, 0)

                    # run inference
                    prediction_scores = model.predict(np_img_crop)
                    prediction_str = label_map[np.argmax(prediction_scores, axis=1)[0]]
                    print(prediction_str)

                    # add to action count dict
                    label_dict[prediction_str] += 1

                    # update normalised
                    total_detected = np.sum(list(label_dict.values()))
                    for kv in label_dict_norm.keys():
                        label_dict_norm[kv] = label_dict[kv]/total_detected
                    #print(list(label_dict_norm.values()))
                    #print(np.sum(list(label_dict_norm.values())))
                    ###################################################

                    # Plot bbox in original image
                    img1 = ImageDraw.Draw(img)
                    img1.rectangle([obj.rectangle.x,
                                    obj.rectangle.y,
                                    obj.rectangle.x + obj.rectangle.w,
                                    obj.rectangle.y + obj.rectangle.h], outline="red", width=1)  # pixels

                    # Add text label
                    img1.text((obj.rectangle.x,
                               obj.rectangle.y),
                              prediction_str,
                              font=ImageFont.truetype("arial", 30),
                              fill=(255, 0, 0))

                    ####
                    #im1.paste(im2)

    # # figure
    # plt.clf()
    # #plt.subplots_adjust(left=0.125, right=0.900, top=0.880, bottom=0.110)
    # plt.subplot(1, 2, 1)
    # plt.tight_layout()
    # plt.imshow(img)
    #
    # # Barplot and save
    # # plot together
    # plt.subplot(1, 2, 2)
    # plt.tight_layout()
    # plt.bar(np.arange(len(label_dict_norm.keys())), [label_dict_norm[kj] for kj in label_map],
    #         color='b')
    # plt.xticks(np.arange(len(label_dict_norm.keys())), label_map, rotation=45)
    # plt.ylim(0, 1)



    # Save
    # Get image as array
    #img = Image.open(local_image)
    img.save(os.path.join(labelled_frames_folder,local_img))

    # plt.savefig(os.path.join(labelled_frames_folder, "frame_plt_" + local_img))
    count += 1




