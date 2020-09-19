'''
import comet_ml in the top of your file
'''
from comet_ml import Experiment
'''
Add the following code anywhere in your machine learning file
'''
experiment = Experiment(api_key="4UcX6KXPQcJ8xWYzOZBu0gHwL",
                        project_name="nmf-clean", workspace="kunalmgupta")


from experiments.train import train_AE, train_SVR

hyp = {
    'learning_rate':1e-4,                                           # Initial learning rate
    'batch_size': 300,                                               # Batch size used for training
    'num_workers':100,                                                # Number of workers used for data loading
    'weight_decay':0.98,                                            # Weight decay used during training
    'num_epochs': 1000,                                              # Number of epochs to train
    'is_small': False,                                               # Set to True if want to work with a small dataset for debug/demo purposes
    'model_folder_SVR':'./train_models_svr/',                               # PATH to where the models are saved during training 
    'points_path': '/kunal-data/NMF_points/',                       # PATH to the directory containing Shapenet points dataset
    'img_path': '/kunal-data2/ShapeNetRendering/'        # PATH to the directory containing ShapeNet renderings from Choy et.al (3dr2n2)
}

train_SVR(experiment, hyp)
# train_SVR(None, hyp)


