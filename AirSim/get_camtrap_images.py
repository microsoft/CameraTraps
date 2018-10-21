# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

#import setup_path 
import airsim

import pprint
import os
import time
import itertools
import numpy as np
import json

def set_segmentation_ids(client, animal_class, num_animals):
    #load animal class name lookup
    animal_lookup = json.load(open('animal_lookup.json','r'))
    #set segmentation values for everything to 0
    success = client.simSetSegmentationObjectID(".*", 0, True)
    #set segmentation for each animal to a different value
    for idx in range(num_animals):
         success = client.simSetSegmentationObjectID(str(animal_lookup[str(animal_class)])+'.*'+str(idx), idx+1, True)

def connect_to_airsim():
    
    connected = False
    while not connected:
        timeout = False
        try:
            client = airsim.VehicleClient()
            client.confirmConnection()
        except:
            timeout = True
        if not timeout:
            connected = True
    return client             


def get_camtrap_images(env_num, animal_class, num_animals):
    output_folder = r"C:\Users\t-sabeer\Documents\AirSimImages\\"

    pp = pprint.PrettyPrinter(indent=4)

    client = connect_to_airsim()

    object_lookup = set_segmentation_ids(client, animal_class, num_animals)


    time.sleep(5) #env needs time to generate
    camera_position_list = [(-45,-45,0,-.1,0.8),(-45,0,0,-.1,-.7),(0,-45,0,-.1,2.0),(0,0,0,-.1,-2.3)]
    for cam_num, cam_pos in enumerate(camera_position_list):
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(cam_pos[0], cam_pos[1], cam_pos[2]), airsim.to_quaternion(cam_pos[3], 0, cam_pos[4])), True)
        #client.simSetCameraOrientation("0", airsim.to_quaternion(-0.161799, 0, 0)); #radians
        #pose = client.simGetVehiclePose()
        #pp.pprint(pose)
        #print('Pose ' + str(cam_num))
        #print(pose)
        time.sleep(1) #new camera position needs time to update
        for x in range(3): # create sequence of 3

            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
                

            for i, response in enumerate(responses):
                if response.pixels_as_float:
                    #print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
                    airsim.write_pfm(os.path.normpath(output_folder + str(env_num) + '_cam_' + str(cam_num) + '_frame_' + str(x) + "_" + str(i) + '.pfm'), airsim.get_pfm_array(response))
                if response.compress:
                    #print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
                    airsim.write_file(os.path.normpath(output_folder + str(env_num) + '_cam_' + str(cam_num) + '_frame_' + str(x) + "_" + str(i) + '.jpg'), response.image_data_uint8)
                else:
                    img = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
                    img_rgba = img.reshape(response.height, response.width, 4)
                    img_rgba = np.flipud(img_rgba) #original image is flipped vertically
                    airsim.write_png(os.path.normpath(output_folder + str(env_num) + '_cam_' + str(cam_num) + '_frame_' + str(x) + "_" + str(i) + '.png'), img_rgba) #write to png 

            time.sleep(1) #1fps frame rate    
        #pose = client.simGetVehiclePose()
        #pp.pprint(pose)
    
    return
# currently reset() doesn't work in CV mode. Below is the workaround
#client.simSetPose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)
