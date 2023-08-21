import cv2
import os
from tqdm import tqdm
import argparse
from pipeline.pipeline import End2End
from pipeline.FaceDetector.mtcnn import MTCNNdetector
from pipeline.FaceDetector.dlib import DLIBdetector
from pipeline.FaceDetector.retina_face import RetinaFaceDetector 
from pipeline.FaceRecognizer.inception_resnet_v1 import ResNetEmb
from pipeline.FaceRecognizer.ArcFace import ArcFaceEmb
from pipeline.FaceAligner import FaceAligner
from config import config, update_config

FACE_DETECTORS = {
    'mtcnn': MTCNNdetector,
    'dlib': DLIBdetector,
    'retina_face': RetinaFaceDetector
}

FACE_RECOGNIZERS = {
    'facenet': ResNetEmb,
    'arcface': ArcFaceEmb
}

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='face verification')
    parser.add_argument("--create_facebank", help="whether perform creating the facebank", action="store_true")
    parser.add_argument("--face_detector", help="name of face detector", type=str, default='retina_face',
                        choices=["retina_face", "mtcnn", "dlib"])
    parser.add_argument("--face_recognizer", help="name of face recognizer", type=str, default='arcface',
                       choices=["facenet", "arcface"])
    parser.add_argument("--path_data", help="Root path for testing images", type=str)
    parser.add_argument("--path_output", help="Root path for resulting images", type=str)
    args = parser.parse_args()
    
    config = update_config(config, args.face_detector, args.face_recognizer)
    detector = FACE_DETECTORS[args.face_detector](config)
    recognizer = FACE_RECOGNIZERS[args.face_recognizer](config)
    aligner = FaceAligner(config)
    e2e = End2End(detector, aligner, recognizer, config)
    
    if args.create_facebank:
        e2e.prepare_facebank(config['path_facebank_images'], config['path_facebank'])
    else:
        e2e.load_facebank(config['path_facebank'])
    
    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)
        
    img_names = os.listdir(args.path_data)
    for img_name in tqdm(img_names):
        img = cv2.imread(os.path.join(args.path_data, img_name))
        img = img[:,:,::-1]
        img_res = e2e.get_image_res(img)
        img_res = img_res[:,:,::-1]
        cv2.imwrite(os.path.join(args.path_output, img_name), img_res)
    
    