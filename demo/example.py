import requests
import json



# #example for image upload
files = {'media': open('S1_F10_R1_PICT0104.JPG', 'rb')}
upload_result = requests.post('http://52.183.35.7/v1/camera_trap_api/detect', files=files).json()



#example for image url
image_url = {'image_url': 'https://awionline.org/sites/default/files/styles/art/public/ealert/image/AWI-eAlert-lion-flickr-StevenTan.jpg?itok=vNFM7zeS'}
url_result = requests.get('http://52.183.35.7/v1/camera_trap_api/detect', params=image_url).json()


