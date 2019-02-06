from flask import Flask, Response, jsonify
from flask import render_template, request, session, url_for, redirect
from flask_assets import Environment, Bundle
from flask_restful import Resource, Api

from . import app, photos, api
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
import time


@app.route('/')
def index():
    return render_template('index.html')
            
@app.route('/upload')
def upload():
    return render_template('upload.html')

#note currently login code only works in staging site
def is_logged_in():
    if 'logged in' in session:
        print('logged in')
        return True
    else:
        return False

def redirect_to_login():
     path =  str(request.url_rule)
     
     if(path == "/"):
          path = "index"

     session['path'] = path

     login_url = os.path.dirname(request.url) + "/login"
     resp = Response(status=307)
     resp.headers['location'] = login_url
     return resp

@app.route("/login")
def login():
    
    REDIRECT_URI = os.path.dirname(request.url) + "/authorized"
    print(REDIRECT_URI)

    auth_state = str(uuid.uuid4())
    session['state'] = auth_state
    authorization_url = app.config["TEMPLATE_AUTHZ_URL"].format(
        aad.TENANT,
        aad.CLIENT_ID,
        REDIRECT_URI,
        auth_state,
        aad.RESOURCE)
    
    resp = Response(status=307)
    resp.headers['location'] = authorization_url
    return resp

@app.route("/authorized")
def authorized():

    REDIRECT_URI = os.path.dirname(request.url) + "/authorized"

    code = request.args['code']
    state = request.args['state']

    if state != session['state']:
        raise ValueError("State does not match")
    
    auth_context = adal.AuthenticationContext(app.config["AUTHORITY_URL"])
    token_response = auth_context.acquire_token_with_authorization_code(code, REDIRECT_URI, aad.RESOURCE,
                                                                        aad.CLIENT_ID, aad.CLIENT_SECRET)
    
    # It is recommended to save this to a database when using a production app.
    session['access_token'] = token_response['accessToken']
    session['logged_in'] = True
    
    template_path = session['path'] + ".html"
    
    #resp = Response(status=307)
    resp.headers['location'] =  os.path.dirname(request.url) + "/" + template_path
    #return resp
    return render_template(template_path)


@app.route('/processurlimage', methods=['GET'])
def processurlimage():
    print('here')
    image_output = []
    download_json = {}
    image_url = request.args.get('image_url')
    org_url = image_url
    filename = image_url.split('/')[-1].split('?')[0]
    replacefilename = filename
    r = requests.get(image_url,
                 stream=True, headers={'User-agent': 'Mozilla/5.0'})
    if r.status_code == 200:
        r.raw.decode_content = True
        image_url = BytesIO(r.raw.read())


    params = {filename: image_url}
    detection_output = requests.post(app.config['API_URL'], files=params).json()
    print(detection_output)
    
    name, ext = os.path.splitext(filename.split('/')[-1])
    replace_characters = [' ', '_', ',', '-']
    for rc in replace_characters:
        name = name.replace(rc, '')
        replacefilename = replacefilename.replace(rc, '')
    
    detection_key = name + '.jpg'
    img_file = detection_output[detection_key].get('img_file')
    #outputFileName = "{}{}".format('static/results/' + name, '.png')
    outputfile = detection_output[detection_key].get('img_file')
    image_output.append({
        "path": outputfile,
        "num_objects": detection_output[detection_key].get('number_of_animals'),
        "org_path": org_url,
        "image_name": filename,
        "result": detection_output[detection_key].get('status'),
        "bboxes": detection_output[detection_key].get('boxes')
    })
    download_json[img_file.split('/')[-1]] = {
        "bboxes": '"' + str(detection_output[detection_key].get('boxes')) + '"',
            "message": detection_output[detection_key].get('status'),
            "num_objects": detection_output[detection_key].get('number_of_animals')

    }

    return render_template('results.html', result_det=image_output, output_json=download_json)


@app.route('/processimages', methods=['POST'])
def processimages():
    image_output = []
    download_json = {}
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            file.filename = file.filename.replace(' ', '')
            

            if os.path.exists(os.path.join(os.getcwd(), 'static/uploads/', file.filename)):
                os.remove(os.path.join(os.getcwd(), 'static/uploads/', file.filename))

            filename = photos.save(
                file,
                name=file.filename
            )
            img_file = photos.url(filename)
            org_path = img_file
            base_url = request.base_url.replace('processimages', 'static/uploads/')
            base_url = base_url  + file.filename


            start = time.time()
            img = Image.open(BytesIO(urlopen.urlopen(img_file).read()))
            img.thumbnail((img.size[0] / 2, img.size[1] / 2), Image.ANTIALIAS)
            os.remove(os.path.join(os.getcwd(), 'static/uploads/', file.filename))
            img.save(os.path.join(os.getcwd(), 'static/uploads/', file.filename))
            end = time.time()
            print('Time taken', end - start)
            files = {img_file.split('/')[-1]: BytesIO(urlopen.urlopen(base_url).read())}
            form = {'image_name': img_file.split('/')[-1]}
            detection_output = requests.post(app.config['API_URL'], files=files).json()
            print(detection_output)
            name, ext = os.path.splitext(img_file.split('/')[-1])
            replace_characters = [' ', '_', ',', '-']
            for rc in replace_characters:
                name = name.replace(rc, '')

            detection_key = name + '.jpg'
            
            
            outputfile = detection_output[detection_key].get('img_file')
            
            image_output.append({
                "path": outputfile,
                "num_objects": detection_output[detection_key].get('number_of_animals'),
                "org_path": org_path,
                "image_name": file.filename,
                "result": detection_output[detection_key].get('status'),
                "bboxes": detection_output[detection_key].get('boxes')
            })
            download_json[img_file.split('/')[-1]] = {
                "bboxes": '"' + str(detection_output[detection_key].get('boxes')) + '"',
                 "message": detection_output[detection_key].get('status'),
                 "num_objects": detection_output[detection_key].get('number_of_animals')

            }

        session['image_output'] = image_output
        session['download_json'] = download_json
        return 'Uploading ....'


@app.route('/results')
def results():
    print('results...')
    # redirect to home if no images to display
    if "image_output" not in session or session['image_output'] == []:
        return redirect(url_for('upload'))

    image_output = session['image_output']
    session.pop('image_output', None)

    return render_template('results.html', result_det=image_output, output_json=session['download_json'])


@app.route('/gallery')
def gallery():
    gallery_images = os.listdir('static/gallery/')
    gallery_images = ['static/gallery/' + img for img in gallery_images]
    gallery_images = random.sample(gallery_images, 12)
    return render_template('gallery.html', gallery_images=gallery_images)



@app.route('/gallery_results/<img_index>', methods=['GET'])
def gallery_results(img_index):
    gallery_images = os.listdir('static/gallery/')
    gallery_images.remove(img_index)
    gallery_images.insert(0, img_index)
    gallery_images = ["/static/gallery/" + img for img in gallery_images]
    output_img = []
    output_json = {}
    for index, img_file in enumerate(gallery_images):
        print('Processing image {} of {}'.format(index,len(gallery_images)))
        with open('static/gallery_results/results.json', 'r') as res:
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

