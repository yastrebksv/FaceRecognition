import dlib
import cv2

class DLIBdetector:
    def __init__(self, cfg):
        self.model = dlib.get_frontal_face_detector()
        
    def run(self, image):
        res = []
        gray = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2GRAY)
        faces = self.model(gray,1)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            res.append([x1,y1,x2,y2])
        return(res)    
        