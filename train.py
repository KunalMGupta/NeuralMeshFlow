import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

from dataset.dataset import get_dataloader
from model.model import NeuralMeshFlow, PointsSVR

def train_AE(experiment, opt):
    '''
    This function trains NMF Auto-encoding task
    experiemt: The comel_ml object where training stats will be stored
    opt: config containing various hyperparameters and other options for experiment
    '''

    # Initialize dataloader and model
    print("*********** INITIALIZATION  ************")
    torch.cuda.set_device(0)  # Choose correct CUDA enabled device
    encoder_type = 'point'    # For AE task, we will choose a Point encoder
    
    training_generator = get_dataloader(encoder_type, opt)   # Choose appropriate training dataset generator
    validation_generator = get_dataloader(encoder_type, opt, split = 'val')   # Choose appropriate training dataset generator

    # **** BUILD NMF MODEL ******
    model = nn.DataParallel(NeuralMeshFlow(encoder_type = encoder_type, zdim=opt.latent_len, time=opt.toi, tol=opt.tolerance)).cuda()

    # Setup training helpers
    print("*********** SETUP TRAINING  ************")
    optimizer = optim.Adam(list(model.parameters()), lr = opt.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=opt.weight_decay)
    os.makedirs(opt.model_folder,exist_ok=True)
    print("Models are being saved at :", opt.model_folder)
    if experiment is None: 
        print("Comet_ml logging is disabled, printing on terminal instead")
    print("*********** BEGIN TRAINING  ************")
    step=0
    for epoch in range(opt.num_epochs):
        for input,_,_ in training_generator:   # Point Cloud, model category, model name is given by training generator
            
            optimizer.zero_grad()
            input = input.cuda()              

            pred0, pred1, pred2, face = model(input)   # Point prediction after each deform block and face information (refer figure 4 in paper)
            
            '''
            **** NOTE ******
            
            To compute Chamfer distances, we will differentiably construct meshes using point predictions pred1, pred2 and the face information. 
            This is made easily possible by Ravi et.al, Pytorch3D. 
            We compute vertex only Chamfer Distance for pred0
            Refer Section 3 (Loss function) in paper for more details.
            
            '''
            
            # Differentiable meshes M_p1, M_p2 (See figure 4)
            mesh_p1 = Meshes(verts = pred1, faces = face)
            mesh_p2 = Meshes(verts = pred2, faces = face)
        
            # Differentiably sample random points from mesh surfaces
            pts1 = sample_points_from_meshes(mesh_p1,num_samples=2562)
            pts2 = sample_points_from_meshes(mesh_p2,num_samples=2562)

            # Compute losses w.r.t. ground truth mesh
            loss1,_ = chamfer_distance(pred0, input)
            loss2,_ = chamfer_distance(pts1,  input)
            loss3,_ = chamfer_distance(pts2,  input)

            loss = (  opt.training_weights[0]*loss1 
                    + opt.training_weights[1]*loss2 
                    + opt.training_weights[2]*loss3)/(
                      opt.training_weights[0]+opt.training_weights[1]+opt.training_weights[2])
            
            loss.backward()
            optimizer.step()
            
            # Weight decay after every 250 steps. Stop once learning rate is too low
            if (step%250 == 0 and scheduler.get_lr()[0] > 5e-7):
                scheduler.step()
                

            if experiment is not None:
                # Log experiment data on comet_ml if requested.
                experiment.log_metric("Total_Loss", loss.item(), step=step)
                experiment.log_metric("Loss_1", loss1.item(), step=step)
                experiment.log_metric("Loss_2", loss2.item(), step=step)
                experiment.log_metric("Loss_3", loss3.item(), step=step)
                experiment.log_metric("Learning rate", scheduler.get_lr()[0]*10**4, step=step)
                
            else:
                # in the absence of comet_ml log, simply do std. out
                print("#Epoch:", epoch+1, 
                      "#Step: ", step+1,
                      "LT: {:.6f}".format(loss.item()), 
                      "L1: {:.6f}".format(loss1.item()),
                      "L2: {:.6f}".format(loss2.item()),
                      "L3: {:.6f}".format(loss3.item()),
                      "lr: {:.4f}".format(scheduler.get_lr()[0]*10**4))
                
            
            step+=1
            
        if epoch%5==0:
            valid_loss = validate_training_AE(validation_generator, model)
            
            if experiment is not None:
                # Log experiment data on comet_ml if requested.
                experiment.log_metric("Valid_Loss", valid_loss, step=step-1)

            else:
                # in the absence of comet_ml log, simply do std. out
                print("#Epoch:", epoch+1, 
                      "#Step: ", step-1,
                      "Valid Loss: {:.6f}".format(valid_loss))
                
            
                
        print("Saving models after #Epoch :", epoch+1)  # Save models after every epoch
        torch.save(model.state_dict(),opt.model_folder+'epoch_{}'.format(epoch))
        
def validate_training_AE(validation_generator, model):
    '''
    This function is used to calculate validation loss during training NMF AE
    '''
    print("Validating model......")
    with torch.no_grad():
        total_loss = 0
        items = 0
        for input,_,_ in validation_generator: 
            input = input.cuda()  
            _, _, pred2, face = model(input)   # Point prediction after each deform block and face information (refer figure 4 in paper)
            mesh_p2 = Meshes(verts = pred2, faces = face)  # Construct Differentiable mesh M_p2
            pts2 = sample_points_from_meshes(mesh_p2,num_samples=2562)  # Differentiably sample random points from mesh surfaces
            loss,_ = chamfer_distance(pts2,  input)
            total_loss+=loss.item()
            items+=1
            
    return total_loss/items   # Return average validation loss

def train_SVR(experiment, opt):
    '''
    This function trains PointsSVR for NMF single view reconstruction task
    experiemt: The comel_ml object where training stats will be stored
    opt: config containing various hyperparameters and other options for experiment
    '''
    # Initialize dataloader and model
    print("*********** INITIALIZATION  ************")
    torch.cuda.set_device(0)  # Choose correct CUDA enabled device
    encoder_type = 'image'    # For SVR task, we will choose an Image encoder
    
    training_generator = get_dataloader(encoder_type, opt)   # Choose appropriate training dataset generator
    validation_generator = get_dataloader(encoder_type, opt, split = 'val')   # Choose appropriate training dataset generator
    model = nn.DataParallel(PointsSVR()).cuda()
    # Setup training helpers
    print("*********** SETUP TRAINING  ************")
    optimizer = optim.Adam(list(model.parameters()), lr = opt.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=opt.weight_decay)
    os.makedirs(opt.model_folder_SVR,exist_ok=True)
    print("Models are being saved at :", opt.model_folder_SVR)
    if experiment is None: 
        print("Comet_ml logging is disabled, printing on terminal instead")
    print("*********** BEGIN TRAINING  ************")
    step=0
    for epoch in range(opt.num_epochs):
        for input,gtpt,_,_ in training_generator:   # Image, Point Cloud, model category, model name is given by training generator
            optimizer.zero_grad()
            input = input.cuda()    
            gtpt = gtpt.cuda()
            predpt = model(input) # Predict a spare point cloud 
            loss,_ = chamfer_distance(predpt, gtpt)
            loss.backward()
            optimizer.step()
            
            # Weight decay after every 500 steps. Stop once learning rate is too low
            if (step%500 == 0 and scheduler.get_lr()[0] > 5e-7):
                scheduler.step()
                
            if experiment is not None:
                # Log experiment data on comet_ml if requested.
                experiment.log_metric("Total_Loss", loss.item(), step=step)
                experiment.log_metric("Learning rate", scheduler.get_lr()[0]*10**4, step=step)
                
            else:
                # in the absence of comet_ml log, simply do std. out
                print("#Epoch:", epoch+1, 
                      "#Step: ", step+1,
                      "LT: {:.6f}".format(loss.item()), 
                      "lr: {:.4f}".format(scheduler.get_lr()[0]*10**4))
            step+=1
            
        if epoch%5==0:
            valid_loss = validate_training_SVR(validation_generator, model)
            
            if experiment is not None:
                # Log experiment data on comet_ml if requested.
                experiment.log_metric("Valid_Loss", valid_loss, step=step-1)

            else:
                # in the absence of comet_ml log, simply do std. out
                print("#Epoch:", epoch+1, 
                      "#Step: ", step-1,
                      "Valid Loss: {:.6f}".format(valid_loss))
            
        print("Saving models after #Epoch :", epoch+1)  # Save models after every epoch
        torch.save(model.state_dict(),opt.model_folder_SVR+'epoch_{}'.format(epoch))
        
def validate_training_SVR(validation_generator, model):
    '''
    This function is used to calculate validation loss during training PointsSVR
    '''
    print("Validating model......")
    with torch.no_grad():
        total_loss = 0
        items = 0
        for input,gtpt,_,_ in validation_generator:   # Image, Point Cloud, model category, model name is given by training generator
            input = input.cuda()    
            gtpt = gtpt.cuda() 
            predpt = model(input) # Predict a spare point cloud
            loss,_ = chamfer_distance(predpt, gtpt)
            total_loss+=loss.item()
            items+=1
            
    return total_loss/items   # Return average validation loss


if __name__ == '__main__':
    from config import get_config
    
    experiment, opt = get_config()
    
    if opt.train == 'AE':
        train_AE(experiment, opt)
        
    elif opt.train == 'SVR':
        train_SVR(experiment, opt)
    
    