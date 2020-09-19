import torch
import numpy as np
import trimesh
from dataset.dataset import *
from model.model import *
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

hyp = {
    'tolerance':1e-5,                                               # tolerance for the ODE Solver (refer ablation in Supplementary)
    'ToI': 0.2,                                                     # Time of integration for each NODE block (refer discussion in Supplementary)
    'latent_len':1000,                                              # Length of the latent embedding 
    'learning_rate':1e-4,                                           # Initial learning rate
    'training_weights':[1,2,7],                                     # Weights corresponding to L_v, L_p1, L_p2
    'batch_size': 50,                                               # Batch size used for training
    'num_workers':10,                                                # Number of workers used for data loading
    'weight_decay':0.98,                                            # Weight decay used during training
    'num_epochs': 150,                                              # Number of epochs to train
    'is_small': False,                                               # Set to True if want to work with a small dataset for debug/demo purposes
    'model_folder':'./train_models/',                               # PATH to where the models are saved during training 
    'points_path': '/kunal-data/NMF_points/',                       # PATH to the directory containing Shapenet points dataset
    'img_path': '/experiments/kunal/DATA/ShapeNetRendering/',       # PATH to the directory containing ShapeNet renderings from Choy et.al (3dr2n2)
    'PATH_svr': './train_models_svr/epoch_370',                     # PATH where trained weights for PointsSVR are stored.
    'PATH_ae': './train_models/epoch_149',                           # PATH where trained weights for NMF AE are stored.
    'generate_ae': '/kunal-data/generate_nmf/points/',
    'generate_svr': '/kunal-data/generate_nmf/svr/'
}


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
        
     
def generate_svr(hyp):
    '''
    This function generates meshes from images using pretrained weights
    hyp: config file
    '''
    
    with torch.no_grad():
        torch.cuda.set_device(0)
        testing_generator = get_dataloader('image', hyp, split ='test')
        model = nn.DataParallel(NeuralMeshFlow(encoder_type = 'image',PATH_svr=hyp['PATH_svr'], zdim=1000, time=0.2)).cuda()
        load_partial_pretrained(model, hyp['PATH_ae'])
        model.eval()
        
        print(" **** Starting evaluation for Images ****")
        for input, _, cat, modelfile in tqdm(testing_generator):
            input = input.cuda()
            _,_, vertices, face = model(input)
            save(hyp['generate_svr'], cat, modelfile, vertices, face)
            
def generate_ae(hyp):
    '''
    This function generates meshes from point clouds using pretrained weights
    hyp: config file
    '''
    
    with torch.no_grad():
#         torch.cuda.set_device(0)
        testing_generator = get_dataloader('point', hyp, split ='test')
        model = nn.DataParallel(NeuralMeshFlow(encoder_type = 'point',PATH_svr=hyp['PATH_ae'], zdim=1000, time=0.2)).cuda()
        load_partial_pretrained(model, hyp['PATH_ae'])
        model.eval()
        
        print(" **** Starting evaluation for Images ****")
        for input, cat, modelfile in tqdm(testing_generator):
            input = input.cuda()
            _,_, vertices, face = model(input)
            save(hyp['generate_ae'], cat, modelfile, vertices, face)
            
            
            
generate_svr(hyp)
#generate_ae(hyp)            