'''
runapp.py

Starts running a web application for labeling samples.
'''
import argparse, bottle, itertools, json, psycopg2, sys, time
import numpy as np
from peewee import *
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

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
    This seems necessary for CORS to work.
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
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host to bind to.')## default='localhost', help='Web server host to bind to.')
    parser.add_argument('--port', type=int, default=8080, help='Web server port port to listen on.')
    parser.add_argument('--verbose', type=bool, default=True, help='Enable verbose debugging.')
    parser.add_argument('--db_name', type=str, default='missouricameratraps', help='Name of Postgres DB with target dataset tables.')
    parser.add_argument('--db_user', type=str, default=None, help='Name of user accessing Postgres DB.')
    parser.add_argument('--db_password', type=str, default=None, help='Password of user accessing Postgres DB.')
    parser.add_argument('--db_query_limit', default=3000, type=int, help='Maximum number of records to read from the Postgres DB.')
    parser.add_argument('--crop_dir', type=str, required=True, help='Path to directory containing cropped images to display.')
    parser.add_argument('--class_list', type=str, required=True, help='Path to .txt file containing classes in dataset.')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to directory where checkpoints will be stored.')
    parser.add_argument('--classifier_checkpoint', type=str, default='', help='Path to a specific classifier checkpoint to load initially.')
    parser.add_argument('--embedding_checkpoint', type=str, default='/home/lynx/pretrainedmodels/embedding_triplet_resnet50_1499/triplet_resnet50_1499.tar', help='Path to a specific embedding checkpoint to load initially.')
    args = parser.parse_args(sys.argv[1:])

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
    checkpoint = load_checkpoint(args.embedding_checkpoint)
    if checkpoint['loss_type'].lower() == 'center' or checkpoint['loss_type'].lower() == 'softmax':
        embedding_net = SoftmaxNet(checkpoint['arch'], checkpoint['feat_dim'], checkpoint['num_classes'], False)
    else:
        embedding_net = NormalizedEmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)
    model = torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])

    # ---------------------------------------------------------------------- #
    # CREATE QUEUE OF IMAGES TO LABEL
    # ---------------------------------------------------------------------- #
    dataset_query = (Detection
                    .select(Detection.id, Detection.category_id, Detection.kind, Detection.category_confidence, Detection.bbox_confidence, Image.file_name, Image.grayscale)
                    .join(Image, on=(Image.id == Detection.image))
                    .order_by(fn.Random())
                    .limit(args.db_query_limit))
    dataset = SQLDataLoader(args.crop_dir, query=dataset_query, is_training=False, kind=DetectionKind.ModelDetection.value, num_workers=8)
    
    grayscale_values = [rec[6] for rec in dataset.samples]
    grayscale_indices = list(itertools.compress(range(len(grayscale_values)), grayscale_values))    # records with grayscale images
    color_indices = list(set(range(len(dataset.samples))) - set(grayscale_indices))                 # records with color images
    detection_conf_values = [rec[4] for rec in dataset.samples]
    dataset.updateEmbedding(model)
    dataset.embedding_mode()
    dataset.train()
    

    kwargs = {}
    kwargs["N"] = 25
    kwargs["already_selected"] = set()
    if args.classifier_checkpoint is not '':
        print('loading pre-trained classifier')
        kwargs["model"] = joblib.load(args.classifier_checkpoint)
        classifier_trained = True
        sampler = get_AL_sampler('confidence')(dataset.em, dataset.getalllabels(), 1234)

        # Use classifier to generate predictions
        dataset.set_kind(DetectionKind.ModelDetection.value)
        X_pred = dataset.em[dataset.current_set]
        y_pred = kwargs["model"].predict(X_pred)
        
        # # Update model predicted class in PostgreSQL database
        # for pos in range(len(y_pred)):
        #     idx = dataset.current_set[pos]
        #     det_id = dataset.samples[idx][0]
        #     matching_detection_entries = (Detection
        #                                 .select(Detection.id, Detection.category_id)
        #                                 .where((Detection.id == det_id)))
        #     mde = matching_detection_entries.get()
        #     command = Detection.update(category_id=y_pred[pos]).where(Detection.id == mde.id)
        #     command.execute()

        # Alternative: batch update PostgreSQL database
        # timer = time.time()
        # det_ids = [dataset.samples[dataset.current_set[pos]][0] for pos in range(len(y_pred))]
        # y_pred = [int(y) for y in y_pred]
        # det_id_pred_pairs = list(zip(det_ids, y_pred))
        # case_statement = Case(Detection.id, det_id_pred_pairs)
        # command = Detection.update(category_id=case_statement).where(Detection.id.in_(det_ids))# switch where and update?
        # command.execute()
        # print('Updating the database the other way took %0.2f seconds'%(time.time() - timer))

        # Update dataset dataloader
        for pos in range(len(y_pred)):
            idx = dataset.current_set[pos]
            sample_data = list(dataset.samples[idx])
            sample_data[1] = y_pred[pos]
            dataset.samples[idx] = tuple(sample_data)
    else:
        kwargs["model"] = MLPClassifier(alpha=0.0001)
        classifier_trained = False
        sampler = get_AL_sampler('uniform')(dataset.em, dataset.getalllabels(), 1234)
    
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
        return bottle.static_file(filename, root='static')
    
    @webUIapp.route('/<filename:re:.*.JPG>')
    def send_image(filename):
        return bottle.static_file(filename, root='/')
        # return bottle.static_file(filename, root='../../../../../../../../../.')## missouricameratraps
        # return bottle.static_file(filename, root='../../../../../../../../../../../.')
    
    ## dynamic routes
    @webUIapp.route('/getClassList', method='POST')
    def get_class_list():
        data = bottle.request.json
        class_list = [cname for cname in open(args.class_list, 'r').read().splitlines()]
        data['class_list'] = class_list
        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)


    @webUIapp.route('/refreshImagesToDisplay', method='POST')
    def refresh_images_to_display():
        '''
        Updates which images are allowed to be sampled by the dataset sampler when the selectors for 
        detection confidence threshold, grayscale images, or class prediction are applied in the webUI, 
        as well as when new images are requested.

        NOTE: This is done by refreshing the list of "already_selected" samples given to the dataset sampler's
        select_batch_ function.
        '''
        global grayscale_indices
        global color_indices
        global detection_conf_values
        global dataset
        global kwargs

        data = bottle.request.json
        
        indices_to_exclude = set() # records that should not be shown
        indices_to_exclude.update(set(dataset.set_indices[DetectionKind.UserDetection.value])) # never show records that have been labeled by the user
        indices_to_exclude.update(set(dataset.set_indices[DetectionKind.ConfirmedDetection.value])) # never show records that have been confirmed by the user
        detection_conf_thresh_indices = [i for i, e in enumerate(detection_conf_values) if e < float(data['detection_threshold'])] # find records below the detection confidence threshold
        indices_to_exclude.update(set(detection_conf_thresh_indices))
        # if data['display_grayscale']:
        #     indices_to_exclude.update(set(color_indices))
        # elif not data['display_grayscale']:
        #     indices_to_exclude.update(set(grayscale_indices))
        
        if data['display_class'] == 'All Species':
            print('Displaying all species')
            pass
        else:
            cat_name = data['display_class'].lower()
            existing_category_entries = {cat.name: cat.id for cat in Category.select()}
            cat_id = existing_category_entries[cat_name]
            dataset_class_labels = [dataset.samples[i][1] for i in range(len(dataset.samples))]
            other_classes = [i for i, cl in enumerate(dataset_class_labels) if cl!=cat_id]
            indices_to_exclude.update(set(other_classes))
            
        kwargs["already_selected"] = indices_to_exclude

        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)

    @webUIapp.route('/loadImages', method='POST')
    def load_images():
        '''
        Returns a batch of images from the dataset sampler to be displayed in the webUI.
        '''
        global dataset
        global sampler
        global kwargs
        global indices

        data = bottle.request.json        
        kwargs["N"] = data['num_images']
        indices_to_exclude = set() # records that should not be shown
        indices_to_exclude.update(set(dataset.set_indices[DetectionKind.UserDetection.value])) # never show records that have been labeled by the user
        indices_to_exclude.update(set(dataset.set_indices[DetectionKind.ConfirmedDetection.value])) # never show records that have been confirmed by the user
        kwargs["already_selected"].update(indices_to_exclude)
        try:
            indices = sampler.select_batch(**kwargs)
            data['success_status'] = True
        except:
            data['success_status'] = False
        data['classifier_trained'] = classifier_trained
        data['display_images'] = {}
        data['display_images']['image_ids'] = [dataset.samples[i][0] for i in indices]
        data['display_images']['image_file_names'] = [dataset.samples[i][5] for i in indices]
        data['display_images']['detection_kinds'] = [dataset.samples[i][2] for i in indices]
        data['display_images']['detection_categories'] = []
        for i in indices:
            if str(dataset.samples[i][1]) == 'None':
                data['display_images']['detection_categories'].append('None')
            else:
                existing_category_entries = {cat.id: cat.name for cat in Category.select()}
                cat_name = existing_category_entries[dataset.samples[i][1]].title()
                data['display_images']['detection_categories'].append(cat_name)


        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)
    
    @webUIapp.route('/loadImagesWithPrediction', method='POST')
    def load_images_with_prediction():
        '''
        Returns a batch of images from the dataset sampler with a specified predicted class to be displayed in the webUI.
        '''
        global dataset
        global sampler
        global kwargs
        global indices

        data = bottle.request.json        
        kwargs["N"] = data['num_images']
        indices = sampler.select_batch(**kwargs)

        # data['display_images'] = {}
        # data['display_images']['image_ids'] = [dataset.samples[i][0] for i in indices]
        # data['display_images']['image_file_names'] = [dataset.samples[i][5] for i in indices]
        # data['display_images']['detection_kinds'] = [dataset.samples[i][2] for i in indices]

        # data['display_images']['detection_categories'] = []
        # for i in indices:
        #     if str(dataset.samples[i][1]) == 'None':
        #         data['display_images']['detection_categories'].append('None')
        #     else:
        #         existing_category_entries = {cat.id: cat.name for cat in Category.select()}
        #         cat_name = existing_category_entries[dataset.samples[i][1]].replace("_", " ").title()
        #         data['display_images']['detection_categories'].append(cat_name)


        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)
    
    @webUIapp.route('/assignLabel', method='POST')
    def assign_label():
        '''
        Assigns a label to a set of images, commits this change to the PostgreSQL database,
        and updates the dataset dataloader accordingly.
        '''
        global dataset
        global indices

        print('Started assignLabel call')

        data = bottle.request.json

        images_to_label = [im['id'] for im in data['images']]
        label_to_assign = data['label']

        # Use image ids in images_to_label to get the corresponding dataset indices
        indices_to_label = []
        indices_detection_ids = [dataset.samples[i][0] for i in indices]
        for im in images_to_label:
            pos = indices_detection_ids.index(im)
            ind = indices[pos]
            indices_to_label.append(ind)

        label_category_name = label_to_assign.lower()
        if label_category_name == 'empty':
            # Update records in dataset dataloader but not in the PostgreSQL database
            moveRecords(dataset, DetectionKind.ModelDetection.value, DetectionKind.UserDetection.value, indices_to_label)
            # numLabeled = len(dataset.set_indices[DetectionKind.UserDetection.value])
        else:
            # Get the category id for the assigned label
            existing_category_entries = {cat.name: cat.id for cat in Category.select()}
            try:
                label_category_id = existing_category_entries[label_category_name]
            except:
                print('The label was not found in the database Category table')
                raise NotImplementedError
            
            # Update entries in the PostgreSQL database
            ## get Detection table entries corresponding to the images being labeled 
            matching_detection_entries = (Detection
                        .select(Detection.id, Detection.category_id)
                        .where((Detection.id << images_to_label))) # << means IN
            ## update category_id, category_confidence, and kind of each Detection entry in the PostgreSQL database      
            for mde in matching_detection_entries:
                command = Detection.update(category_id=label_category_id, category_confidence=1, kind=DetectionKind.UserDetection.value).where(Detection.id == mde.id)
                command.execute()
            
            # Update records in dataset dataloader
            for il in indices_to_label:
                sample_data = list(dataset.samples[il])
                sample_data[1] = label_category_id
                sample_data[2] = DetectionKind.UserDetection.value
                sample_data[3] = 1
                dataset.samples[il] = tuple(sample_data)
            moveRecords(dataset, DetectionKind.ModelDetection.value, DetectionKind.UserDetection.value, indices_to_label)
            # print(set(dataset.set_indices[4]).update(set(indices_to_label)))
            dataset.set_indices[4] = list(set(dataset.set_indices[4]).union(set(indices_to_label))) # add the index to the set of labeled/confirmed indices
            # numLabeled = len(dataset.set_indices[DetectionKind.UserDetection.value])
            print([len(x) for x in dataset.set_indices])

        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)

    @webUIapp.route('/confirmPredictedLabel', method='POST')
    def confirm_predicted_label():
        global dataset
        global indices

        data = bottle.request.json

        image_to_label = data['image']
        label_to_assign = data['label']
        
        # Use image id images_to_label to get the corresponding dataset index
        indices_detection_ids = [dataset.samples[i][0] for i in indices]
        pos = indices_detection_ids.index(image_to_label)
        index_to_label = indices[pos]

        label_category_name = label_to_assign.lower()
        if label_category_name == 'empty':
            # Update records in dataset dataloader but not in the PostgreSQL database
            moveRecords(dataset, DetectionKind.ModelDetection.value, DetectionKind.ConfirmedDetection.value, [index_to_label])
            # numLabeled = len(dataset.set_indices[DetectionKind.UserDetection.value]) # userdetection + confirmed detection?
        else:
            # Get the category id for the assigned label
            existing_category_entries = {cat.name: cat.id for cat in Category.select()}
            try:
                label_category_id = existing_category_entries[label_category_name]
            except:
                print('The label was not found in the database Category table')
                raise NotImplementedError
            
            # Update entries in the PostgreSQL database
            ## get Detection table entries corresponding to the images being labeled 
            matching_detection_entries = (Detection
                        .select(Detection.id, Detection.category_id)
                        .where((Detection.id==image_to_label))) # << means IN
            ## update category_id, category_confidence, and kind of each Detection entry in the PostgreSQL database      
            mde = matching_detection_entries.get()
            command = Detection.update(category_id=label_category_id, category_confidence=1, kind=DetectionKind.ConfirmedDetection.value).where(Detection.id == mde.id)
            command.execute()
            
            # Update records in dataset dataloader
            sample_data = list(dataset.samples[index_to_label])
            sample_data[1] = label_category_id
            sample_data[2] = DetectionKind.ConfirmedDetection.value
            sample_data[3] = 1
            dataset.samples[index_to_label] = tuple(sample_data)
            moveRecords(dataset, DetectionKind.ModelDetection.value, DetectionKind.ConfirmedDetection.value, [index_to_label])
            dataset.set_indices[4] = list(set(dataset.set_indices[4]).union({index_to_label})) # add the index to the set of labeled/confirmed indices
            # numLabeled = len(dataset.set_indices[DetectionKind.UserDetection.value])
            print([len(x) for x in dataset.set_indices])
        
        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)
    
    @webUIapp.route('/trainClassifier', method='POST')
    def train_classifier():
        global dataset
        global kwargs
        global sampler
        global classifier_trained
        global X_pred
        global y_pred

        data = bottle.request.json

        # Train on samples that have been labeled so far
        # dataset.set_kind(DetectionKind.UserDetection.value)
        dataset.set_kind(4)
        print(dataset.current_set)
        print(type(dataset.current_set))
        X_train = dataset.em[dataset.current_set]
        y_train = np.asarray(dataset.getlabels())
        # print(y_train)
        timer = time.time()
        kwargs["model"].fit(X_train, y_train)
        print('Training took %0.2f seconds'%(time.time() - timer))

        timer = time.time()
        joblib.dump(kwargs["model"], "%s/%s_%04d.skmodel"%(args.checkpoint_dir, 'classifier', len(dataset.current_set)))
        print('Saving classifier checkpoint took %0.2f seconds'%(time.time() - timer))

        
        # Predict on the samples that have not been labeled
        timer = time.time()
        dataset.set_kind(DetectionKind.ModelDetection.value)
        X_pred = dataset.em[dataset.current_set]
        y_pred = kwargs["model"].predict(X_pred)
        print('Predicting on unlabeled samples took %0.2f seconds'%(time.time() - timer))
        # print(y_pred)

        # Update model predicted class in PostgreSQL database
        # timer = time.time()
        # for pos in range(len(y_pred)):
        #     idx = dataset.current_set[pos]
        #     det_id = dataset.samples[idx][0]
        #     matching_detection_entries = (Detection
        #                                 .select(Detection.id, Detection.category_id)
        #                                 .where((Detection.id == det_id)))
        #     mde = matching_detection_entries.get()
        #     command = Detection.update(category_id=y_pred[pos]).where(Detection.id == mde.id)
        #     command.execute()
        # print('Updating the database took %0.2f seconds'%(time.time() - timer))

        # Alternative: batch update PostgreSQL database
        # timer = time.time()
        # det_ids = [dataset.samples[dataset.current_set[pos]][0] for pos in range(len(y_pred))]
        # y_pred = [int(y) for y in y_pred]
        # det_id_pred_pairs = list(zip(det_ids, y_pred))
        # case_statement = Case(Detection.id, det_id_pred_pairs)
        # command = Detection.update(category_id=case_statement).where(Detection.id.in_(det_ids))
        # command.execute()
        # print('Updating the database the other way took %0.2f seconds'%(time.time() - timer))

        # Update dataset dataloader
        timer = time.time()
        for pos in range(len(y_pred)):
            idx = dataset.current_set[pos]
            sample_data = list(dataset.samples[idx])
            sample_data[1] = y_pred[pos]
            dataset.samples[idx] = tuple(sample_data)
        print('Updating the dataset dataloader took %0.2f seconds'%(time.time() - timer))
        
        if not classifier_trained:
            # once the classifier has been trained the first time, switch to AL sampling
            classifier_trained = True
            sampler = get_AL_sampler('confidence')(dataset.em, dataset.getalllabels(), 1234)

        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)
    
    @webUIapp.route('/showFullsizeImage', method='POST')
    def show_fullsize_image():
        data = bottle.request.json
        
        image_src = data['img_src']
        
        matching_image_entries = (Image
                                .select(Image.file_name, Image.source_file_name)
                                .where((Image.file_name == image_src)))
        try:
            mie = matching_image_entries.get()
            data['success_status'] = True
            data['fullsize_src'] = mie.source_file_name
        except:
            data['success_status'] = False
        
        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)
    
    @webUIapp.route('/getSequentialImages', method='POST')
    def get_sequential_images():
        data = bottle.request.json
        
        image_src = data['img_src']
        
        matching_image_entries = (Image
                                .select(Image.seq_id, Image.seq_num_frames, Image.frame_num)
                                .where((Image.file_name == image_src)))
        try:
            mie = matching_image_entries.get()
            if mie.seq_num_frames > 1:
                images_in_seq = (Image
                                .select(Image.source_file_name)
                                .where((Image.seq_id == mie.seq_id))
                                .order_by(Image.frame_num))
                image_sequence = sorted(list(set([i.source_file_name for i in images_in_seq])))
                if len(image_sequence) > 10:
                    minidx = max(mie.frame_num - 4, 0)
                    maxidx = min(mie.frame_num + 4, len(image_sequence))
                    image_sequence = image_sequence[minidx:maxidx+1]
                data['image_sequence'] = image_sequence
            data['success_status'] = True
        except:
            data['success_status'] = False
        
        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)

    webUIapp.run(**webUIapp_server_kwargs)