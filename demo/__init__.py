from flask import Flask
from flask_assets import Environment
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

from flask_restful import Resource, Api
# from . import aadConfig as aad

import os

app = Flask(__name__)
app.debug = True
api = Api(app)

webassets = Environment(app)
dropzone = Dropzone(app)

app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

#api url
app.config['API_URL'] = ''
app.config['API_RESULTS_URL'] = ''


# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'
app.config['DROPZONE_DEFAULT_MESSAGE'] = 'DRAG OR BROWSE<br/><br/>UP TO 10 IMAGES'
app.config['DROPZONE_TIMEOUT'] = 30000 * 60 * 10
app.config['DROPZONE_PARALLEL_UPLOADS'] = 10
app.config['DROPZONE_MAX_FILE_SIZE'] = 10
app.config['CHECK_LOGIN'] = False
# app.config['AUTHORITY_URL'] =  aad.AUTHORITY_HOST_URL + '/' + aad.TENANT
app.config['TEMPLATE_AUTHZ_URL']= ('https://login.microsoftonline.com/{}/oauth2/authorize?' +
                      'response_type=code&client_id={}&redirect_uri={}&' +
                      'state={}&resource={}')
app.config["REDIRECT_URI"] = ''
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

# app.config['DROPZONE_IN_FORM'] = True
# app.config['DROPZONE_UPLOAD_ON_CLICK'] = True 
# app.config['DROPZONE_UPLOAD_BTN_ID'] =  'submit'
# app.config[' DROPZONE_UPLOAD_ACTION'] = 'processimages'


# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(os.getcwd() , 'static', 'uploads')
app.config['UPLOADED_PHOTOS_ALLOW'] = set(['png', 'jpg', 'jpeg', 'JPG'])



photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


#model configuration
app.config['MODEL_FILE'] = 'checkpoint/frozen_inference_graph.pb'
app.config['OUTPUT_FOLDER'] = 'static/results/'


from . import assets
