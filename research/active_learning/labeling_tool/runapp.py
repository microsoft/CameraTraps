'''
runapp.py

Starts running a web application for labeling samples.
'''
import argparse, bottle, json, psycopg2, sys
import numpy as np
from peewee import *
from sklearn.neural_network import MLPClassifier

sys.path.append('../')
from Database.DB_models import *
from DL.sqlite_data_loader import SQLDataLoader
from DL.networks import *
from DL.losses import *
from DL.utils import *
from DL.Engine import Engine

from sampling_methods.constants import get_AL_sampler
from sampling_methods.constants import get_wrapper_AL_mapping
get_wrapper_AL_mapping()

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


#--------some stuff needed to get AJAX to work with bottle?--------#
def enable_cors():
    '''
    From https://gist.github.com/richard-flosi/3789163
    This globally enables Cross-Origin Resource Sharing (CORS) headers for every response from this server.
    '''
    bottle.response.headers['Access-Control-Allow-Origin'] = '*'
    bottle.response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    bottle.response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

def do_options():
    '''
    This seems necessary for CORS to work
    '''
    bottle.response.status = 204
    return

def moveRecords(dataset, srcKind, destKind, rList):
    for e in rList:
        if e in dataset.set_indices[srcKind]:
            dataset.set_indices[srcKind].remove(e)
            dataset.set_indices[destKind].append(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a web user interface for labeling camera trap images for classification.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host to bind to.')
    parser.add_argument('--port', type=int, default=8080, help='Web server port port to listen on.')
    parser.add_argument('--verbose', type=bool, default=True, help='Enable verbose debugging.')
    parser.add_argument('--db_name', type=str, default='missouricameratraps', help='Name of Postgres DB with target dataset tables.')
    parser.add_argument('--db_user', type=str, default=None, help='Name of user accessing Postgres DB.')
    parser.add_argument('--db_password', type=str, default=None, help='Password of user accessing Postgres DB.')
    parser.add_argument('--db_query_limit', default=50000, type=int, help='Maximum number of records to read from the Postgres DB.')
    args = parser.parse_args(sys.argv[1:])

    args.crop_dir = "/home/lynx/data/missouricameratraps/crops/" ## TODO: hard coding this for now, but should not actually need this since Image table stores paths to crops
    args.strategy = 'confidence' ## TODO: hard coding this for now

    # -------------------------------------------------------------------------------- #
    # PREPARE TO QUEUE IMAGES FOR LABELING
    # -------------------------------------------------------------------------------- #
    
    ## Connect as USER to database DB_NAME through peewee and initialize database proxy
    DB_NAME = args.db_name
    USER = args.db_user
    PASSWORD = args.db_password
    target_db = PostgresqlDatabase(DB_NAME, user=USER, password=PASSWORD, host='localhost')
    target_db.connect(reuse_if_open=True)
    db_proxy.initialize(target_db)

    ## Load embedding model
    checkpoint = load_checkpoint("/home/lynx/pretrainedmodels/embedding_triplet_resnet50_1499/triplet_resnet50_1499.tar")
    if checkpoint['loss_type'].lower() == 'center' or checkpoint['loss_type'].lower() == 'softmax':
        embedding_net = SoftmaxNet(checkpoint['arch'], checkpoint['feat_dim'], checkpoint['num_classes'], False)
    else:
        embedding_net = NormalizedEmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)
    model = torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    
    # ##
    # dataset_query = Detection.select(Detection.id, Detection.category, Detection.kind, Image.file_name).join(Image, on=(Image.id == Detection.image)).order_by(fn.random()).limit(50) ## TODO: should this really be order_by random?
    # # dataset_query = Detection.select(Detection.id, Oracle.label, Detection.kind).join(Oracle, on=(Oracle.detection == Detection.id)).order_by(fn.random()).limit(50) ## TODO: should this really be order_by random?
    # dataset = SQLDataLoader(args.crop_dir, query=dataset_query, is_training=False, kind=DetectionKind.ModelDetection.value, num_workers=8)
    # # print(list(dataset_query.tuples()))
    # dataset.updateEmbedding(model)
    # dataset.embedding_mode()
    # dataset.train()
    # sampler = get_AL_sampler(args.strategy)(dataset.em, dataset.getalllabels(), 12)
    # numLabeled = len(dataset.set_indices[DetectionKind.UserDetection.value])
    # print(numLabeled)

    # kwargs = {}
    # kwargs["N"] = 10#args.active_batch
    # kwargs["already_selected"] = dataset.set_indices[DetectionKind.UserDetection.value]
    # kwargs["model"] = MLPClassifier(alpha=0.0001)
    
    # indices = np.random.choice(dataset.current_set, kwargs["N"], replace=False).tolist()
    # print(indices)

    # assert 2==3, 'break here'
    

    # -------------------------------------------------------------------------------- #
    # CREATE AND SET UP A BOTTLE APPLICATION FOR THE WEB UI
    # -------------------------------------------------------------------------------- #
    
    webUIapp = bottle.Bottle()
    webUIapp.add_hook("after_request", enable_cors)
    webUIapp_server_kwargs = {
        "server": "tornado",
        "host": args.host,
        "port": args.port
    }
    
    ## static routes (to serve CSS, etc.)
    @webUIapp.route('/')
    def index():
        return bottle.static_file("index.html", root='static/html')
    
    @webUIapp.route('/favicon.ico')
    def favicon():
        return
    
    @webUIapp.route('/<filename:re:js\/.*>')
    def send_js(filename):
        return bottle.static_file(filename, root='static')
    
    @webUIapp.route('/<filename:re:css\/.*>')
    def send_css(filename):
        return bottle.static_file(filename, root='static')
    
    @webUIapp.route('/<filename:re:img\/placeholder.JPG>')
    def send_placeholder_image(filename):
        # print('trying to load image', filename)
        return bottle.static_file(filename, root='static')
    
    @webUIapp.route('/<filename:re:.*.JPG>')
    def send_image(filename):
        # print('trying to load camtrap image', filename)
        return bottle.static_file(filename, root='../../../../../../../../../.')
    
    ## dynamic routes
    @webUIapp.route('/refreshImageDataset', method='POST')
    def refresh_image_dataset():
        global dataset_query
        global dataset
        global sampler

        data = bottle.request.json

        # ---------------------------------------------------------------------- #
        # CREATE QUEUE OF IMAGES TO LABEL
        # ---------------------------------------------------------------------- #
        dataset_query = (Detection
                        .select(Detection.id, Detection.category, Detection.kind, Image.file_name)
                        .join(Image, on=(Image.id == Detection.image))
                        .where((Detection.bbox_confidence >= data['detection_threshold']) & (Image.grayscale == data['display_grayscale']))
                        .order_by(fn.Random())
                        .limit(args.db_query_limit))
        # print(list(dataset_query.tuples()))
        dataset = SQLDataLoader(args.crop_dir, query=dataset_query, is_training=False, kind=DetectionKind.ModelDetection.value, num_workers=8)
        dataset.updateEmbedding(model)
        dataset.embedding_mode()
        dataset.train()
        # sampler = get_AL_sampler(args.strategy)(dataset.em, dataset.getalllabels(), 12)
        sampler = get_AL_sampler('uniform')(dataset.em, dataset.getalllabels(), 12)
        print('SUCCESSFULLY REFRESHED THE IMAGE DATASET')

        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)


    @webUIapp.route('/loadImages', method='POST')
    def load_images():
        global dataset_query
        global dataset
        global sampler
        global indices

        data = bottle.request.json
        
        # # TODO: return file names of crops to show from "totag" csv or database

        # # ---------------------------------------------------------------------- #
        # # CREATE QUEUE OF IMAGES TO LABEL
        # # ---------------------------------------------------------------------- #
        # dataset_query = (Detection
        #                 .select(Detection.id, Detection.category, Detection.kind, Image.file_name)
        #                 .join(Image, on=(Image.id == Detection.image))
        #                 .where((Detection.bbox_confidence >= data['detection_threshold']) & (Image.grayscale == data['display_grayscale']))
        #                 .limit(args.db_query_limit))
        # dataset = SQLDataLoader(args.crop_dir, query=dataset_query, is_training=False, kind=DetectionKind.ModelDetection.value, num_workers=8)
        # dataset.updateEmbedding(model)
        # dataset.embedding_mode()
        # dataset.train()
        # sampler = get_AL_sampler(args.strategy)(dataset.em, dataset.getalllabels(), 12)

        numLabeled = len(dataset.set_indices[DetectionKind.UserDetection.value])
        
        kwargs = {}
        kwargs["N"] = data['num_images']
        kwargs["already_selected"] = dataset.set_indices[DetectionKind.UserDetection.value] # TODO: change this to be those images that have already been shown to the user, rather than those the user has already labeled
        kwargs["model"] = MLPClassifier(alpha=0.0001)
        
        if numLabeled < 100:
            indices = np.random.choice(dataset.current_set, kwargs["N"], replace=False).tolist()
        else:
            indices = sampler.select_batch(**kwargs)
        # print(indices)
        # print(list(dataset_query.tuples())[indices])
        # print([dataset_query[i] for i in indices])

        # existing_image_entries = (Image
        #                         .select(Image.id, Image.file_name, Detection.kind, Detection.category)
        #                         .join(Detection, on=(Image.id == Detection.image))
        #                         .where((Image.grayscale == data['display_grayscale']) & (Detection.bbox_confidence >= data['detection_threshold']))
        #                         .order_by(fn.Random()).limit(data['num_images']))

        data['display_images'] = {}
        data['display_images']['image_ids'] = [dataset_query[i].id for i in indices]
        # data['display_images']['image_ids'] = [ie.id for ie in dataset_query]
        data['display_images']['image_file_names'] = [dataset_query[i].image.file_name for i in indices]
        # data['display_images']['image_file_names'] = [ie.image.file_name for ie in dataset_query]
        data['display_images']['detection_kinds'] = [dataset_query[i].kind for i in indices]
        # data['display_images']['detection_kinds'] = [ie.kind for ie in dataset_query]
        data['display_images']['detection_categories'] = [str(dataset_query[i].category) for i in indices]
        # data['display_images']['detection_categories'] = [str(ie.category) for ie in dataset_query]

        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)
    
    @webUIapp.route('/assignLabelDB', method='POST')
    def assign_label():
        global dataset_query
        global dataset
        global indices

        data = bottle.request.json

        # Get the category id for the assigned label
        label_to_assign = data['label']
        label_category_name = label_to_assign.lower().replace(" ", "_")
        existing_category_entries = {cat.name: cat.id for cat in Category.select()}
        try:
            label_category_id = existing_category_entries[label_category_name]
        except:
            print('The label was not found in the database Category table')
            raise NotImplementedError

        images_to_label = [im['id'] for im in data['images']]
        
        # Get the Detection table entries corresponding to the images being labeled 
        ## NOTE: detection id (and image_id) are the same as image id in missouricameratraps_test
        matching_detection_entries = (Detection
                                .select(Detection.id, Detection.category_id)
                                .where((Detection.id << images_to_label))) # << means IN

        # Update the category_id, category_confidence, and kind of each Detection entry        
        for mde in matching_detection_entries:
            command = Detection.update(category_id=label_category_id, category_confidence=1, kind=DetectionKind.UserDetection.value).where(Detection.id == mde.id)
            command.execute()
        
        # Get the dataset indices corresponding to the images being labeled
        labeled_indices = []
        dataset_query_indices_records = [dataset_query[i].id for i in indices]
        for im in images_to_label:
            pos = dataset_query_indices_records.index(im)
            ind = indices[pos]
            labeled_indices.append(ind)
        
        # Update records in dataset
        moveRecords(dataset, DetectionKind.ModelDetection.value, DetectionKind.UserDetection.value, labeled_indices)
        numLabeled = len(dataset.set_indices[DetectionKind.UserDetection.value])
        
        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)
    
    @webUIapp.route('/trainClassifier', method='POST')
    def train_classifier():
        global dataset_query
        global dataset

        dataset_query = (Detection
                        .select(Detection.id, Detection.category, Detection.kind, Image.file_name)
                        .join(Image, on=(Image.id == Detection.image))
                        .order_by(fn.Random())
                        .limit(args.db_query_limit))
        dataset = SQLDataLoader(args.crop_dir, query=dataset_query, is_training=False, kind=DetectionKind.ModelDetection.value, num_workers=8)
        dataset.updateEmbedding(model)
        dataset.embedding_mode()

        data = bottle.request.json

        # Train on samples that have been labeled so far
        dataset.set_kind(DetectionKind.UserDetection.value)
        X_train = dataset.em[dataset.current_set]
        y_train = np.asarray(dataset.getlabels())
        print(y_train)

        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)
    
    webUIapp.run(**webUIapp_server_kwargs)