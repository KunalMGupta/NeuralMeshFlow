import torch
from torch.utils import data
import torch.nn as nn
import json
import random
import os
import imageio
import numpy as np

def get_dataloader(type, opt, split = 'train', is_small=False):
    '''
    Helper function let's you choose a dataloader
    type: Choose from 'point' or 'image' for shape completion or SVR tasks.
    opt: options for choosing dataloader
    split: Choose from 'train', 'test' and 'val'
    points_path: PATH to the directory containing Shapenet points dataset
    img_path: PATH to the directory containing ShapeNet renderings from Choy et.al (3dr2n2)
    is_small: Set to True if wish to work with a small dataset of size 100. For demo/debug purpose
    '''
    
    # Parameters
    params = {'batch_size': opt.batch_size,
              'shuffle': True,
              'num_workers': opt.num_workers,
              'drop_last' : True}
    
    if split == 'test' or split =='val':
        params['shuffle'] = False
        
    training_set = Dataset(split, opt,  encoder_type = type, is_small=opt.is_small)
    training_generator = data.DataLoader(training_set, **params)

    print("Dataloader for {}s with Batch Size : {} and {} workers created for {}ing.".format(type, params['batch_size'], params['num_workers'], split))
    return training_generator


def read_color_image(filename):
    return np.array(imageio.imread(filename))[...,:3]


def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x 

class Dataset(data.Dataset):
    '''
    Main dataset class used for testing/training. Can be used for both SVR and shape completion/AE tasks. 
    '''
    
    def __init__(self, split_type, opt, encoder_type='points', is_small=False):
        '''
        Initialization function.
        split_type: 'train', 'valid', 'test' used to specify the partion of dataset to be used
        opt: options for choosing dataloader
        encoder_type: 'points' or 'image' used to specify the type of input data to be used. Will fetch appropriate image data if 'image' is used.
        is_small: Set to True if wish to work with a small dataset of size 100. For demo/debug purpose
        '''
        
        # Load the file containing model splits. Splits are made based on 3dr2n2 by Choy et.al. 
        
        with open('/kunal-data/NeuralMeshFlow/dataloader/mysplit.json', 'r') as outfile:  
            split = json.load(outfile)

        self.dataset_dir     = opt.points_path     # PATH to points dataset
        self.dataset_dir_img = opt.img_path        # PATH to Image dataset
        self.models = []                              # Stores all models
        self.model2cat = {} # stores models with corresponding category
        self.count = 0
        self.split = split_type
        
        mycats = list(split[split_type].keys()) # Categories to be used for training/testing (based on split)
        
        for cat in mycats:
            models = list(split[split_type][cat])
            for model in models:
                self.models.append(model)       # All models used for training
                self.model2cat[model] = cat
        

        random.shuffle(self.models)   # Randomly shuffle the models

        if is_small:            # Work with a very small dataset (used for debug/demo)
            print(" !!! Using small dataset of size 100 !!!  ")
            self.models = self.models[:100]
        
        self.encoder_type = encoder_type
        
        print("total models for {}ing : ".format(self.split), len(self.models))
        
        
    def __len__(self):
        return len(self.models)
    
    def __get_imagenum__(self):
        '''
        Helper function to fetch the correct image number (00, 01, .. 22, 23) for .png files. 
        Selects randomly for training. For testing, chooses numbers 00->23 as is called. 
        Thus 23 rounds are required to get all images for testing.
        '''
        if self.split == 'test':
            
            imagenum = int(self.count/self.__len__())
            self.count+=1

        else:
            imagenum = np.random.randint(0,24)
            
        if imagenum < 10:
            imagenum = '0'+str(imagenum)
        
        return str(imagenum)
    
    def __getitem__(self,index):
        
        modelfile = self.models[index]
        
        
        if self.encoder_type == 'image':
            # Fetch dataset for SVR
            try:
                imagenum = self.__get_imagenum__()  # Fetech image number
                I = read_color_image(self.dataset_dir_img+'{}/{}/rendering/{}.png' \
                                     .format(self.model2cat[modelfile],modelfile, imagenum))  # Read image
                X = np.load(self.dataset_dir+'{}/{}/points.npy'.format(self.model2cat[modelfile],modelfile), allow_pickle=True)  # Read point cloud 
                
            except:
                # Incase fail to load the file, simply return the first file of dataset to prevent crashing.
                
                modelfile = self.models[0]
                imagenum = self.__get_imagenum__()   # Fetech image number
                I = read_color_image(self.dataset_dir_img+'{}/{}/rendering/{}.png' \
                                     .format(self.model2cat[modelfile],modelfile, imagenum))   # Read image
                X = np.load(self.dataset_dir+'{}/{}/points.npy'.format(self.model2cat[modelfile],modelfile), allow_pickle=True)   # Read point cloud 
 
            # Preprocess Image
            I = I.astype(np.float64)/255.0   # Convert image (0-255, int) to (0-1, float)
            I = torch.from_numpy(I)
            I = normalize_imagenet(I).transpose(0,2).float()   # Normalize w.r.t. Imagenet stats and (H,W,3) -> (3,H,W) for adequate training
            
            #Preprocess Point cloud
            X /= np.max(np.linalg.norm(X, axis=1))   # Point cloud to lie withing a unit sphere
            mask = np.random.randint(0,X.shape[0],1024)  # Randomly sample 1024 points for training of PointsSVR network
            X = torch.from_numpy(X[mask,...]).float() # Point cloud now lies within sphere of 0.5 radius
            
            return I, X, self.model2cat[modelfile] , modelfile+'_'+imagenum  # return image, pointcloud with corresponding model category and name
        
        elif self.encoder_type == 'point':
            # Fetch dataset for AE 
            
            try:
                X = np.load(self.dataset_dir+'{}/{}/points.npy'.format(self.model2cat[modelfile],modelfile), allow_pickle=True)   # Read point cloud 
                
            except:
                # Incase fail to load the file, simply return the first file of dataset to prevent crashing.
                
                print("Error loading:", self.model2cat[modelfile], modelfile)
                modelfile = self.models[0]
                X = np.load(self.dataset_dir+'{}/{}/points.npy'.format(self.model2cat[modelfile],modelfile), allow_pickle=True)   
                
            X /= np.max(np.linalg.norm(X, axis=1))   # Point cloud to lie withing a unit sphere
            mask = np.random.randint(0,X.shape[0],2562)  # Randomly sample 2562 points for training NMF AE
            X = torch.from_numpy(X[mask,...]).float() 
            
            return X, self.model2cat[modelfile], modelfile 

