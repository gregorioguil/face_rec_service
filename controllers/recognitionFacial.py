from interactor.recognitionFacialBs import RecognitionFacialBs
#from interactor.recognitionTrainBs import RecognitionTrainBs

class RecognitionFacialController:
    def __init__(self, photo):
        self.photo = photo

    def recognize(self):
        bs = RecognitionFacialBs(self.photo)
        return bs.recognize()

    def train(self):
        bs = RecognitionTrainBs()
        bs.train()
        return