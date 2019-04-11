from flask import Flask, Response, jsonify
from flask import render_template, request, session, url_for, redirect
from flask_assets import Environment, Bundle
from flask_restful import Resource, Api
from urllib.parse import unquote
from werkzeug.utils import secure_filename
from flask import send_from_directory
from requests_toolbelt.multipart import decoder
from . import app, photos, api
import CameraTrapAssets.apiconfig as apiconfig

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
from . import aadConfig as aad
from . import login_helper 
import time
import io

api_url_format = '{0}/camera-trap/detect?confidence={1}&render={1}'
results_folder = '/CameraTrapAssets/results/'


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
    return {'Ocp-Apim-Subscription-Key': apiconfig.api['subscription_key']}

def call_api():
    params = {
      'confidence': 0.8,
      'render': True
    }

    posted_files = request.files
    files = {}
    images = []
    for f in posted_files:
        file = posted_files.get(f)
        image_name = secure_filename(file.filename)
        images.append(image_name)

        if not image_name.lower().endswith('.jpg'):
            continue

        files[image_name] = (image_name, file, 'image/jpeg')
    
    r = requests.post(apiconfig.api['base_url'] + 'detect', params=params, files=files, headers=get_api_headers())
    
    start = time.time()

    for f in posted_files:
        file = posted_files.get(f)
        image_name = secure_filename(file.filename)
        try:
            save_file = file
            save_file.save(os.getcwd() + '/CameraTrapAssets/imgs/' + image_name)
        except Exception as e:
            print(str(e))
    
    end = time.time()
    print(end - start)

    return r, images

#def track_images(file, name):
#   print(str(e))

@app.route('/processImages', methods=['POST'])
def process_images():
    error = False
    print("calling api...")
    
    try:
        
        r, images = call_api()
        results = decoder.MultipartDecoder.from_response(r)

        image_output = []
        for part in results.parts:
            #print(part)
            # part is a BodyPart object with b'Content-Type', and b'Content-Disposition', 
            # the later includes 'name' and 'filename' info
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
        #print(image_output)
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
