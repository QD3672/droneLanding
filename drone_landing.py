import airsim
import cv2
import numpy as np 
import time
from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
import torch
#model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)


#model = YOLO("D:/DroneSimulation/droneLanding/best.pt")


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

drone_pose = client.simGetVehiclePose()
drone_position = client.getMultirotorState().kinematics_estimated.position
print(drone_position)
client.takeoffAsync().join()
all_assets = client.simListAssets()
bp_npc_assets = [asset for asset in all_assets if "BP_NPC" in asset]

client.moveByVelocityBodyFrameAsync(0, 0, 1, 1)
client.simGetCollisionInfo()
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
    #results = model.track(rawImage, show = True, conf = 0.7, iou = 0.5)  # predict on an image
    #print(results[0].boxes)
    pred_depth, confidence, output_dict = model.inference({'input': rawImage})
    pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models
    normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details


'''   
while True:
    take_image()
    
cv2.destroyAllWindows() 
'''
