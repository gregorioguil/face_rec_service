from flask import Flask, request
from flask_cors import CORS
from controllers.detectionFacial import DetectionFacialController
from controllers.recognitionFacial import RecognitionFacialController
from controllers.user import UserController

app = Flask(__name__)
CORS(app)

@app.route('/face/detection', methods = ['POST'])
def detection():
    if request.method == 'POST':
        f = request.files['photo']
        detec = DetectionFacialController(f)
        return detec.detect()
    return 'upload photo fail'

@app.route('/face/recognition', methods = ['POST'])
def recognition():
    if request.method == 'POST':
        f = request.files['photo']
        r = RecognitionFacialController(f)
        return r.recognize()
    return 'upload photo fail'

# @app.route('/recognition/train', methods = ['GET'])
# async def train():
#     if request.method == 'GET':
#         r = RecognitionTrainController()
#         await r.train()
#         return {
#             "train": "waiting"
#         }

@app.route('/user/<id>', methods = ['GET'])
def getUser(id):
    controller = UserController()
    return controller.get(id)

@app.route('/users', methods = ['GET'])
def getUsers():
    controller = UserController()
    response = controller.list()
    return response

@app.route('/user', methods = ['POST'])
def createUser():
    user = request.get_json()
    controller = UserController()
    return controller.create(user)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")