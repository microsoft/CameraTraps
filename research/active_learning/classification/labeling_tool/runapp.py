'''
runapp.py

Starts running a web application for labeling samples.
'''
import argparse, json, psycopg2, sys
import bottle
from peewee import *
sys.path.append('../../Database')
from DB_models import *

# webUIapp = bottle.Bottle()

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



# #--------dynamic routings for the webUIapp bottle server--------#
# @webUIapp.post('/loadImages')
# def load_images():
#     bottle.response.content_type = 'application/json'
#     returned_images = bottle.request.json
#     returned_images = [{"id": "testid", "name": "test name"}]
#     return json.dumps(returned_images)

# webUIapp.run(host='localhost', port=8080)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a web user interface for labeling camera trap images for classification.')
    parser.add_argument('--host', type=str, default='localhost', help='Web server host to bind to.')
    parser.add_argument('--port', type=int, default=8080, help='Web server port port to listen on.')
    parser.add_argument('--verbose', type=bool, default=True, help='Enable verbose debugging.')
    parser.add_argument('--db_name', type=str, default='missouricameratraps', help='Name of Postgres DB with target dataset tables.')
    parser.add_argument('--db_user', type=str, default=None, help='Name of user accessing Postgres DB.')
    parser.add_argument('--db_password', type=str, default=None, help='Password of user accessing Postgres DB.')
    args = parser.parse_args(sys.argv[1:])

    # Create a queue of images to pre-load

    # Create and set up a bottle application for the web UI
    webUIapp = bottle.Bottle()
    webUIapp.add_hook("after_request", enable_cors)
    webUIapp_server_kwargs = {
        "host": args.host,
        "port": args.port
    }
    ## static routes (to serve CSS, etc.)
    @webUIapp.route('/')
    def index():
        return bottle.static_file("index.html", root='static/html')
    
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
    
    # dynamic routes
    @webUIapp.route('/loadImages', method='POST')
    def load_images():
        data = bottle.request.json
        
        # # TODO: return file names of crops to show from "totag" csv or database
        DB_NAME = args.db_name
        USER = args.db_user
        PASSWORD = args.db_password
        #HOST = 'localhost'
        #PORT = 5432

        ## Try to connect as USER to database DB_NAME through peewee
        pretrain_db = PostgresqlDatabase(DB_NAME, user=USER, password=PASSWORD, host='localhost')
        db_proxy.initialize(pretrain_db)
        existing_image_entries = Image.select().where(Image.grayscale == False).order_by(fn.Random()).limit(data['num_images'])

        # for image_entry in existing_image_entries:
        #     print(image_entry.file_name)
        data['file_names'] = [ie.file_name[1:] for ie in existing_image_entries]
        # data['label'] = None

        bottle.response.content_type = 'application/json'
        bottle.response.status = 200
        return json.dumps(data)
    
    webUIapp.run(**webUIapp_server_kwargs)