'''
runapp.py

Starts running a web application for labeling samples.
'''

from bottle import Bottle, static_file, request

webUIapp = Bottle()

class DBMiddleware():
    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)
    
    def getImageBatch(self, limit=None):
        response = {}

        return response



@webUIapp.route('/')
def hello():
    return(static_file("index.html", root='static/html'))

@webUIapp.route('/<filename:re:.*>')
def send_static(filename):
    return static_file(filename, root='static')

@webUIapp.get('/getImageBatch')
def get_image_batch():
    postData = request.body.read()
    dataIDs = postData['imageIDs'] #????
    json = webUImiddleware.getImageBatch(dataIDs)

webUIapp.run(host='localhost', port=8080)