import torch
import numpy as np
import trimesh
from dataset.dataset import *
from model.model import *
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def save(folder, cat, modelfile, vertices, faces):
    '''
    This function is used to store predicted meshes in their category specific folders
    folder: Path to directory where generated meshes should be stored
    cat: list of categories
    modelfile: list of model file names
    vertices: list of vertices for meshes
    faces: list of faces for meshes
    '''
    
    for idx in range(vertices.shape[0]):
        # Extract mesh to CPU using trimesh
        v = vertices[idx,...].cpu().numpy()
        f = faces[idx,...].cpu().numpy()
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        
        os.makedirs('{}/{}'.format(folder,cat[idx]), exist_ok=True)
        mesh.export('{}/{}/{}.obj'.format(folder,cat[idx],modelfile[idx]));
        
     
def generate_svr(opt):
    '''
    This function generates meshes from images using pretrained weights
    opt: config file
    '''
    
    with torch.no_grad():
        torch.cuda.set_device(0)
        model = nn.DataParallel(NeuralMeshFlow(encoder_type = 'image',PATH_svr=opt.pretrained_svr_weights, zdim=1000, time=0.2)).cuda()
        load_partial_pretrained(model, opt.pretrained_ae_weights)
        model.eval()
        
        print(" **** Generating shapes with image input ****")
        for j in range(23):
            testing_generator = get_dataloader('image', opt, split ='test',img_num=j)
            for input, _, cat, modelfile in tqdm(testing_generator):
                input = input.cuda()
                _,_, vertices, face = model(input)
                save(opt.generate_svr, cat, modelfile, vertices, face)
            
def generate_ae(opt):
    '''
    This function generates meshes from point clouds using pretrained weights
    opt: config file
    '''
    
    with torch.no_grad():
#         torch.cuda.set_device(0)
        testing_generator = get_dataloader('point', opt, split ='test')
        model = nn.DataParallel(NeuralMeshFlow(encoder_type = 'point', zdim=1000, time=0.2)).cuda()
        load_partial_pretrained(model, opt.pretrained_ae_weights)
        model.eval()
        
        print(" **** Generating shapes with point cloud input ****")
        for input, cat, modelfile in tqdm(testing_generator):
            input = input.cuda()
            _,_, vertices, face = model(input)
            save(opt.generate_ae, cat, modelfile, vertices, face)
            
            
if __name__ == '__main__':
    from config import get_config
    
    experiment, opt = get_config()
    
    if opt.generate == 'AE': 
        generate_ae(opt)   
    elif opt.generate == 'SVR': 
        generate_svr(opt)
    else:
        print("Invalid generate request")
