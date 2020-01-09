from flask import Flask, Response, jsonify
from flask import render_template, request, session, url_for, redirect
from flask_assets import Environment, Bundle
from flask_restful import Resource, Api
from urllib.parse import unquote
from werkzeug.utils import secure_filename
from flask import send_from_directory
from requests_toolbelt.multipart import decoder
from . import app, photos, api
import apiconfig as apiconfig
from requests_toolbelt import MultipartEncoder

import os
# from . import model 
import json
import requests
import uuid
import adal
import traceback

import random
import urllib.request as urlopen
from io import BytesIO
from PIL import Image
# from . import aadConfig as aad
from . import login_helper 
from log import Log
import time
import io


#api_url = apiconfig.api['base_url'] + '/camera-trap/detect?confidence={1}&render={1}'
results_folder = '/CameraTrapAssets/results/'
upload_folder = '/CameraTrapAssets/uploads'

log = Log()

# routes for cameratrapassets as these are being loaded
# from the cameratrapassets directory instead of the static directory
@app.route('/CameraTrapAssets/img/<path:path>')
def site_images(path):
    return send_from_directory('CameraTrapAssets/img/', path)

@app.route('/CameraTrapAssets/gallery/<path:path>')
def gallery_images(path):
    return send_from_directory('CameraTrapAssets/gallery/', path)

@app.route('/CameraTrapAssets/gallery_results/<path:path>')
def gallery_resut_images(path):
    return send_from_directory('CameraTrapAssets/gallery_results/', path)

@app.route('/CameraTrapAssets/results/<path:path>')
def result_images(path):
    return send_from_directory('CameraTrapAssets/results/', path)

def get_api_headers():
    return {'Ocp-Apim-Subscription-Key': apiconfig.SUBSCRIPTION_KEY}

def save_posted_files():
    images = []
    posted_files = request.files
    for f in posted_files:
        file = posted_files.get(f)
        print(file.filename)
        image_name = secure_filename(file.filename)
        images.append(image_name)
        img_path = os.getcwd() + "/" + upload_folder + "/" + image_name
        print(img_path)
        file.save(img_path)
    
    print("files saved...")
    return images
def resize_images(images):
    img_path = os.getcwd() + "/" + upload_folder + "/" 
    for image_name in images:
        im = Image.open(img_path + image_name)
        width, height = im.size
        new_width = (width * 2) / 3
        new_height = (height * 2) / 3
        im.thumbnail((new_width, new_height), Image.ANTIALIAS)
        im.save(img_path + image_name, "JPEG")
    print("files resized...")

def call_api(images):
    num_images_to_upload = 8

    detection_confidence = 0.8
    render_boxes = True  

    params = {
        'confidence': detection_confidence,
        'render': render_boxes
    }

    files = {}

    num_images = 0
    for image_name in images:
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
    
        if num_images >= num_images_to_upload:
            break
        else:
            num_images += 1
    
        img_path = os.path.join("." + upload_folder, image_name)
        print(img_path)
        files[image_name] = (image_name, open(img_path, 'rb'), 'image/jpeg')

    print('number of images to send:', len(files))

    response = requests.post(apiconfig.BASE_URL + 'detect', params=params, files=files, headers=get_api_headers())

    print(response.status_code)
    print(response.ok)
    
    if not response.ok:
        log.error("\nResponse ok: "+ str(response.ok))
        log.error("Response ok: "+ str(response.reason))
        log.error("Response ok: "+ str(response.text))

        for image_name in images:
            print("images send:")
            print(image_name)

        print(response.reason)
        print(response.text)
        return None
    
    return response

#def track_images(file, name):
#   print(str(e))

@app.route('/processImages', methods=['POST'])
def process_images():
    error = False
    
    try:
        print("uploaded....")
        images = save_posted_files()
        print(images)
        #resize_images(images)
       
        response = call_api(images)

        if response is None:
            return "Error occurred while calling API"

        results = decoder.MultipartDecoder.from_response(response)

        image_output = []
        for part in results.parts:
            headers = {}
            for k, v in part.headers.items():
                headers[k.decode(part.encoding)] = v.decode(part.encoding)
            if headers.get('Content-Type', None) == 'image/jpeg':
                c = headers.get('Content-Disposition')
                image_name = c.split('name="')[1].split('"')[0]  
                image = Image.open(io.BytesIO(part.content))
                image.save(os.getcwd() + results_folder + image_name)
    
            elif headers.get('Content-Type', None) == 'application/json':
                #bbox points, confidence
                json_results = json.loads(part.content.decode())
                print(json_results)
                for img_name in images:
                    img_result = json_results[img_name]
                    #print(img_result)
                    image_output.append({
                        "path": results_folder + img_name,
                        "num_objects": len(img_result),
                        "org_path": results_folder + img_name,
                        "image_name": img_name,
                        "result": {},
                        "bboxes": {}
                    })
        session['image_output'] = image_output
        
    except Exception as e:
        print(str(e))
        error = True
    if error:
        return "Error occurred while calling API"
    return "Success"

@app.route('/')
def index():
    return render_template('index.html')
            
@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/results')
def results():
    print('results...')
    # redirect to home if no images to display
    if "image_output" not in session or session['image_output'] == []:
        return redirect(url_for('upload'))

    image_output = session['image_output']
    session.pop('image_output', None)
    
    return render_template('results.html', result_det=image_output, output_json=[{}])


@app.route('/gallery')
def gallery():
    gallery_images = os.listdir('CameraTrapAssets/gallery/')
    gallery_images = ['CameraTrapAssets/gallery/' + img for img in gallery_images]
    #gallery_images = random.sample(gallery_images, 12)
    return render_template('gallery.html', gallery_images=gallery_images)

@app.route('/gallery_results/<img_index>', methods=['GET'])
def gallery_results(img_index):
    gallery_images = os.listdir('CameraTrapAssets/gallery/')
    gallery_images.remove(img_index)
    gallery_images.insert(0, img_index)
    gallery_images = ["/CameraTrapAssets/gallery/" + img for img in gallery_images]
    output_img = []
    output_json = {}
    for index, img_file in enumerate(gallery_images):
        print('Processing image {} of {}'.format(index,len(gallery_images)))
        with open('CameraTrapAssets/gallery_results/results.json', 'r') as res:
            res_data = json.load(res)
            num_objects = res_data[img_file.split('/')[-1]]['num_objects']
            output_img.append({
                "path": img_file.replace('gallery', 'gallery_results'),
                "num_objects": num_objects ,
                "org_path": img_file,
                "image_name": img_file.split('/')[-1],
                "message": "Animal Detected" if num_objects > 0 else "No Animal Detected",
                "bboxes": res_data[img_file.split('/')[-1]]['bboxes']
                
            })

            output_json[img_file.split('/')[-1]] = {
                "bboxes": '"' + str(res_data[img_file.split('/')[-1]]['bboxes']) + '"',
                 "message": "Animal Detected" if num_objects > 0 else "No Animal Detected",
                 "num_objects": num_objects

            }

        
    return render_template('results.html', result_det=output_img, output_json=output_json)

@app.route('/about')
def about():
	return render_template('about.html')

@app.errorhandler(413)
def page_not_found(e):
    return "Your error page for 413 status code", 413

def ext_lowercase(name):
    base, ext = os.path.splitext(name)
    return base + ext.lower()
