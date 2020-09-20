import argparse

def get_config():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default='AE', type=str, help = "Train 'AE' or 'SVR'")
    parser.add_argument("--generate", default='AE', type=str, help = "Generate for 'AE' or 'SVR'")
    parser.add_argument("--tolerance", default=1e-5, type=float, help = "tolerance for the ODE Solver (refer ablation in Supplementary)")
    parser.add_argument("--toi", default=0.2, type=float, help = "Time of integration for each NODE block (refer discussion in Supplementary)")
    parser.add_argument("--latent_len", default=1000, type=int, help = "Length of the latent embedding")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help = "Initial learning rate")
    parser.add_argument("--training_weights", default=[1,2,7], type=list, help = "Weights corresponding to L_v, L_p1, L_p2")
    parser.add_argument("--batch_size", default=250, type=int, help = "Batch size used for training")
    parser.add_argument("--num_workers", default=125, type=int, help = "Number of workers used for data loading")
    parser.add_argument("--weight_decay", default=0.98, type=float, help = "Weight decay used during training")
    parser.add_argument("--num_epochs", default=150, type=int, help = "Number of epochs to train")
    parser.add_argument("--is_small", default=False, help = "Set to True if want to work with a small dataset for debug/demo purposes")
    parser.add_argument("--model_folder", default='./train_models/', type=str, help = "PATH to where the models are saved during training ")
    parser.add_argument("--points_path", default = './data/ShapeNetPoints/', type=str, help = "PATH to the directory containing Shapenet points dataset")
    parser.add_argument("--img_path", default = './data/ShapeNetRendering/', type=str, help = "PATH to the directory containing Shapenet points dataset")
    parser.add_argument("--model_folder_SVR", default = './train_models_svr/', type=str, help = "PATH to where the models are saved during training SVR")
    parser.add_argument("--generate_ae", default = './generate_nmf/points/', type=str, help = "PATH to where meshes for AE are stored")
    parser.add_argument("--generate_svr", default = './generate_nmf/svr/', type=str, help = "PATH to where meshes for SVR are stored")
    
    parser.add_argument("--comet_API", default = None, type=str, help = "your API for comet_ml workspace")
    parser.add_argument("--comet_workspace", default = None, type=str, help = "your comet_ml workspace name")
    parser.add_argument("--comet_project_name", default = "NeuralMeshFlow", type=str, help = "Name of this project in comet_ml")
    
    opt = parser.parse_args()
    
    if opt.comet_API is not None:
        from comet_ml import Experiment
        
        experiment = Experiment(api_key=opt.comet_API,
                        project_name=opt.comet_project_name, workspace=opt.comet_workspace)
        
    else:
        experiment = None
        
        
    return experiment, opt
    

