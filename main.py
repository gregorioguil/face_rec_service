import asyncio
from flask import Flask, render_template, request
from controllers.detectionFacial import DetectionFacialController
from controllers.recognitionFacial import RecognitionFacialController
#from controllers.recognitionTrain import RecognitionTrainController


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/test')
def test():
    return 'Hello World! Eu estou maluco.'


@app.route('/detection/facial', methods = ['POST'])
def detection():
    if request.method == 'POST':
        f = request.files['photo']
        detec = DetectionFacialController(f)
        return detec.detect()
        #f.save(secure_filename(f.filename))
        #return 'file uploaded successfully'
    return 'upload photo fail'

@app.route('/recognition/facial', methods = ['POST'])
def recognition():
    if request.method == 'POST':
        f = request.files['photo']
        r = RecognitionFacialController(f)
        return r.recognize()

# @app.route('/recognition/train', methods = ['GET'])
# async def train():
#     if request.method == 'GET':
#         r = RecognitionTrainController()
#         await r.train()
#         return {
#             "train": "waiting"
#         }

