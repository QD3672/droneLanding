import airsim
import cv2
import numpy as np 
import time
from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image


model = YOLO("D:/DroneSimulation/droneLanding/best.pt")


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToZAsync(-8,5).join()
# set camera name and image type to request images and detections
camera_name = "3"
image_type = airsim.ImageType.Scene

# set detection radius in [cm]
#client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100) 
def take_image():
    response = client.simGetImages([airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)])[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    if img1d.size != response.width * response.height * 3:
        return
    
    rawImage = img1d.reshape(response.height, response.width, 3)
    results = model.track(rawImage, show = True, conf = 0.7, iou = 0.5)  # predict on an image
    #print(results[0].boxes)

    
while True:
    take_image()
    
cv2.destroyAllWindows() 
