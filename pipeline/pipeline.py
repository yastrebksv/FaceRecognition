import os
import cv2
import torch
from .utils import draw_box_name

class End2End:
    def __init__(self, face_detector, face_aligner, face_embedder, cfg):
        self.face_detector = face_detector
        self.face_aligner = face_aligner
        self.face_embedder = face_embedder
        self.cfg = cfg
        
    def run(self, image):
        bbox_faces, landmarks = self.face_detector.run(image)
        faces = [self.face_aligner.get_face(image, bbox, ldms) for bbox, ldms in zip(bbox_faces, landmarks)]
        embedding_faces = [self.face_embedder.run(face) for face in faces]
        return(bbox_faces, faces, embedding_faces)
    
    def prepare_facebank(self, path_images, path_facebank):
        self.facebank_embeddings = []
        self.facebank_names = []
        name_folders = os.listdir(path_images)
        for name in name_folders:
            path_folder = os.path.join(path_images, name)
            images_name = os.listdir(path_folder)
            embeddings = []
            for img_name in images_name:
                img = cv2.imread(os.path.join(path_folder, img_name))[:,:,::-1]
                _,_,emb = self.run(img)
                embeddings.append(emb[0])
                self.facebank_names.append(name)
            embeddings = torch.cat(embeddings).mean(0,keepdim=True)   
            self.facebank_embeddings.append(embeddings)
        self.facebank_embeddings = torch.cat(self.facebank_embeddings)
        if path_facebank:
            facebank = {'embeddings': self.facebank_embeddings, 
                        'names': self.facebank_names}
            torch.save(facebank, path_facebank)              
    
    def load_facebank(self, path_facebank):
        facebank = torch.load(path_facebank, map_location=self.cfg['device'])
        self.facebank_embeddings = facebank['embeddings']
        self.facebank_names = facebank['names'] 

    def infer_on_image(self, image, threshold):
        bboxes, faces, embs = self.run(image)
        embs = torch.cat(embs)
        diff = embs.unsqueeze(-1) - self.facebank_embeddings.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        min_dists, min_idx = torch.min(dist, dim=1)
        min_idx[min_dists > threshold] = -1
        return min_idx, min_dists, bboxes
    
    def get_image_res(self, image):
        min_idx, min_dists, bboxes = self.infer_on_image(image, 1.5)
        names = [self.facebank_names[idx] if idx != -1 else None for idx in min_idx]
        for i in range(len(bboxes)):
            image = draw_box_name(bboxes[i], names[i], image)
        image = cv2.UMat.get(image)    
        return image    
    
    
    