from interactor.detectionFacialBs import DetectionFacialBs

class DetectionFacialController:
    def __init__(self, photos):
        self.photos = photos

    def detect(self):
        bs = DetectionFacialBs(self.photos)
        return bs.detect()

        