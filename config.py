import torch
from torchvision import transforms as trans

config = {
    'device': "cpu", # torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'use_landms_for_align': True,
    'face_size': 112,
    'path_facebank_images': 'data/images_facebank',
    'path_facebank': 'data/facebank/facebank.pth'
}

def update_config(config, name_detector, name_recognizer):
    if name_detector == 'retina_face':
        config['variance'] = [0.1, 0.2]
        config['min_sizes'] = [[16, 32], [64, 128], [256, 512]]
        config['steps'] = [8, 16, 32]   
        config['clip'] = False
        config['pretrain'] = 'pretrain'    
        config['name'] = 'Resnet50' # TODO add mobilenet
        config['path_resnet_model'] = 'pretrain_models/retina_face/Resnet50_Final.pth'
        config['return_layers'] = {'layer2': 1, 'layer3': 2, 'layer4': 3} # for ResNet50
        config['in_channel'] = 256 # for ResNet
        config['out_channel'] = 256 # for REsNet
        
    elif name_detector == 'mtcnn':
        config['path_pnet'] = 'pretrain_models/mtcnn/pnet.pt'
        config['path_rnet'] = 'pretrain_models/mtcnn/rnet.pt'
        config['path_onet'] = 'pretrain_models/mtcnn/onet.pt'        
        
        
    if name_recognizer == 'facenet':
        config['path_vggface2'] = 'pretrain_models/facenet/20180402-114759-vggface2.pt'
        config['use_landms_for_align'] = False
        config['face_size'] = 160
        
    elif name_recognizer == 'arcface':
        config['path_arcface_model'] = 'pretrain_models/arcface/model_ir_se50.pth'
        config['net_depth'] = 50
        config['drop_ratio'] = 0.6
        config['net_mode'] = 'ir_se'
        config['test_transform'] = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        
        
    return config    
        
        
