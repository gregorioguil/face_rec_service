from flask import Flask, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from controllers.detectionFacial import DetectionFacialController
from controllers.recognitionFacial import RecognitionFacialController
from controllers.user import UserController

app = Flask(__name__)
CORS(app)

model = load_model("models/model.h5")

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
        return r.recognize(model)
    return 'upload photo fail'

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