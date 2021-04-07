from .utils import warp_and_crop_face, get_reference_facial_points, extract_face

class FaceAligner:
    def __init__(self, cfg):
        self.reference = get_reference_facial_points(default_square= True)
        self.cfg = cfg
    
    def get_face(self, image, bbox, landmarks):
        if self.cfg['use_landms_for_align']:
            face = warp_and_crop_face(image, landmarks, self.reference, 
                                      crop_size=(self.cfg['face_size'], self.cfg['face_size']))
        else:
            face = extract_face(image, bbox, self.cfg['face_size'], 0, None)
        return face
        