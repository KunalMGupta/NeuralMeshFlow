# import comet_ml in the top of your file
# from comet_ml import Experiment
    
# Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="4UcX6KXPQcJ8xWYzOZBu0gHwL",
#                         project_name="nmf-clean", workspace="kunalmgupta")


from experiments.train import train_AE

hyp = {
    'tolerance':1e-5,                                               # tolerance for the ODE Solver (refer ablation in Supplementary)
    'ToI': 0.2,                                                     # Time of integration for each NODE block (refer discussion in Supplementary)
    'latent_len':1000,                                              # Length of the latent embedding 
    'learning_rate':1e-5,                                           # Initial learning rate
    'training_weights':[1,2,7],                                     # Weights corresponding to L_v, L_p1, L_p2
    'batch_size': 250,                                               # Batch size used for training
    'num_workers':125,                                                # Number of workers used for data loading
    'weight_decay':0.98,                                            # Weight decay used during training
    'num_epochs': 150,                                              # Number of epochs to train
    'is_small': False,                                               # Set to True if want to work with a small dataset for debug/demo purposes
    'model_folder':'./train_models/',                               # PATH to where the models are saved during training 
    'points_path': '/kunal-data/NMF_points/',                       # PATH to the directory containing Shapenet points dataset
    'img_path': '/experiments/kunal/DATA/ShapeNetRendering/'        # PATH to the directory containing ShapeNet renderings from Choy et.al (3dr2n2)
}

# train_AE(experiment, hyp)
train_AE(None, hyp)


