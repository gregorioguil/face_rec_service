from interactor.recognitionFacialBs import RecognitionFacialBs

class RecognitionFacialController:
    def __init__(self, photo):
        self.photo = photo

    def recognize(self, model):
        bs = RecognitionFacialBs(self.photo, model)
        return bs.recognize()