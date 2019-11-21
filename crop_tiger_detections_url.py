from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time

import requests
from io import BytesIO

# Add your Computer Vision subscription key to your environment variables.
# In console: setx COMPUTER_VISION_SUBSCRIPTION_KEY 250b934f1f574b6ca2ec6900e1480d7b
# setx COMPUTER_VISION_ENDPOINT https://sm-cv.cognitiveservices.azure.com/
# https://docs.microsoft.com/es-es/azure/cognitive-services/cognitive-services-apis-create-account?tabs=multiservice%2Cwindows#configure-an-environment-variable-for-authentication

if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
    subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
else:
    print("\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()
# Add your Computer Vision endpoint to your environment variables.
if 'COMPUTER_VISION_ENDPOINT' in os.environ:
    endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
else:
    print("\nSet the COMPUTER_VISION_ENDPOINT environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()



# Create a client
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))


# Get URL image with different objects
remote_image_url_objects = "https://previews.123rf.com/images/airubon/airubon1708/airubon170800584/84673738-tiger-walking-relax-in-natural.jpg"
# get image
response = requests.get(remote_image_url_objects)
img = Image.open(BytesIO(response.content))
print(img.size) #(600,462)

# Run obj detection (input url only)
detect_objects_results_remote = computervision_client.detect_objects(remote_image_url_objects)


# Print detected objects results with bounding boxes
c = 0 # crop counter
if len(detect_objects_results_remote.objects) == 0:
    print("No objects detected.")
else:
    for obj in detect_objects_results_remote.objects:
        # print object name and bounding box coords
        # print("object {} at location {}, {}, {}, {}".format(obj.object_property, \
        #                                                     obj.rectangle.x, obj.rectangle.x + obj.rectangle.w, \
        #                                                     obj.rectangle.y, obj.rectangle.y + obj.rectangle.h))

        if obj.object_property in ["tiger", "animal", "mammal"]:
            # Crop bbox
            img_crop = img.crop((obj.rectangle.x,
                                 obj.rectangle.y,
                                 obj.rectangle.x + obj.rectangle.w,
                                 obj.rectangle.y + obj.rectangle.h))
            # Save crop
            img_crop.save("figures/crop_{}.jpg".format(c))

            c += 1

        else:
            print(obj.object_property)

