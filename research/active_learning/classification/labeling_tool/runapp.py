'''
runapp.py

Starts running a web application for labeling samples.
'''
import json
import bottle

webUIapp = bottle.Bottle()

# class DBMiddleware():
#     def __init__(self, config):
#         self.config = config
#         self.dbConnector = Database(config)
    
#     def getImageBatch(self, limit=None):
#         response = {}

#         return response


#--------static routings for the webUIapp bottle server--------#
@webUIapp.route('/')
def hello():
    return(bottle.static_file("index.html", root='static/html'))

@webUIapp.route('/<filename:re:.*>')
def send_static(filename):
    return bottle.static_file(filename, root='static')

#--------dynamic routings for the webUIapp bottle server--------#
@webUIapp.post('/loadImages')
def load_images():
    bottle.response.content_type('application/json')
    returned_images = bottle.request.json
    
    returned_images = [{"id": "testid", "name": "test name"}]
    return json.dumps(returned_images)
    # postData = request.body.read()
    # dataIDs = postData['imageIDs'] #????

webUIapp.run(host='localhost', port=8080)