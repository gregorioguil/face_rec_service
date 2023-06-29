from interactor.recognitionFacialBs import RecognitionFacialBs

class RecognitionFacialController:
    def __init__(self, photo):
        self.photo = photo

    def recognize(self):
        bs = RecognitionFacialBs(self.photo)
        return bs.recognize()