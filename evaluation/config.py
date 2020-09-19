def get_configs(METHOD, TYPE):
    '''
    This function is used to get evaluation parameters like data path, etc.
    
    METHOD: str; name of the method used
    TYPE: is it image or point cloud type data.
    '''
    
    assert METHOD in ['AtlasNet-25', 'AtlasNet', 'pixel2mesh', 'meshrcnn','occnet-1','occnet-2','occnet-3','ablation'], 'Please specify correct method for evaluation. Choose from one of these : AtlasNet-25, AtlasNet, pixel2mesh, meshrcnn, occnet-1, occnet-2, occnet-3, ablation'
    
    assert TYPE in ['Images', 'Points', 'ab-1e-2', 'ab-1e-3', 'ab-1e-4', 'no-norm', 'refine-0', 'refine-1'], 'Please specify correct datatype. Choose from one of these : Images, Points or from ablations ab-1e-2, ab-1e-3, ab-1e-4, no-norm, refine-0, refine-1'
    
    '''
    Modify the following paths according to where you saved respective mesh predictions
    '''
    
    # Ground Truth
    GDTH_PATH = '/kunal-data/NMF_points/'            # Path to ground truth points dataset (we evaluate against this)
    # Note: for Pixel2Mesh and MeshRCNN, ground truth meshes are stored with respective predicted meshes
    
    DIR = '/kunal-data2'        # Directory where predicted meshes are stored
    
    # AtlasNet-25
    AT25_PRED_PATH_IMAGES = '{}/new_atlasnet-25/svr/'.format(DIR)   # SVR meshes are stored here
    AT25_OUTFILE_IMAGES =  'results/atlasnet-25-images.json'      # File where evaluation results are stored for SVR
    
    AT25_PRED_PATH_POINTS = '{}/new_atlasnet-25/points/'.format(DIR) # AE meshes are stored here
    AT25_OUTFILE_POINTS = 'results/atlasnet-25-points.json'       # File where evaluation results are stored for AE
    
    # AtlasNet
    AT_PRED_PATH_IMAGES = '{}/new_atlasnet/svr/'.format(DIR)   # SVR meshes are stored here
    AT_OUTFILE_IMAGES =  'results/atlasnet-images.json'      # File where evaluation results are stored for SVR
    
    AT_PRED_PATH_POINTS = '{}/new_atlasnet/points/'.format(DIR) # AE meshes are stored here
    AT_OUTFILE_POINTS = 'results/atlasnet-points.json'       # File where evaluation results are stored for AE
    
    # Pixel2Mesh
    P2M_PRED_PATH_IMAGES = '{}/pixel2mesh_data_new/'.format(DIR)  # SVR meshes are stored here
    P2M_OUTFILE_IMAGES = 'results/pixel2mesh.json'              # File where evaluation results are stored for SVR
    
    # MeshRCNN
    MRCNN_PRED_PATH_IMAGES = '{}/meshrcnn_data_new/'.format(DIR) # SVR meshes are stored here
    MRCNN_OUTFILE_IMAGES = 'results/meshrcnn-smooth.json'    # File where evaluation results are stored for SVR
    
    # OccNet
    OCNN_PRED_PATH_IMAGES = '{}/occnet/occnet-{}/svr/'.format(DIR,METHOD[-1])  # SVR meshes are stored here
    OCNN_OUTFILENAME_IMAGES = 'results/occnet-{}-svr.json'.format(METHOD[-1])# File where evaluation results are stored for SVR
    
    OCNN_PRED_PATH_POINTS = '{}/occnet/occnet-{}/points/'.format(DIR,METHOD[-1])  # AE meshes are stored here
    OCNN_OUTFILE_POINTS = 'results/occnet-{}-points.json'.format(METHOD[-1])  # File where evaluation results are stored for AE
    
    #NMF
    NMF_PRED_PATH_IMAGES = '{}/generate_nmf/svr/'.format(DIR) # SVR meshes are stored here
    NMF_OUTFILE_IMAGES = 'results/nmf-images.json'        # File where evaluation results are stored for SVR
    
    NMF_PRED_PATH_POINTS = '{}/generate_nmf/points/'.format(DIR) # AE meshes are stored here
    NMF_OUTFILE_POINTS = 'results/nmf-points.json'           # File where evaluation results are stored for AE
    
    #Ablations
    ABL_PRED_PATH_POINTS = '{}/ablation/{}/'.format(DIR,TYPE) # Ablation meshes are stored here
    ABL_OUTFILE_POINTS =  'results/{}.json'.format(DIR,TYPE) # File where evaluation results are stored for ablation
    

    print("\n")
    if METHOD == 'AtlasNet-25':
        
        if TYPE == 'Images':
            print("****** Doing for AtlasNet-25 Images*******")
            return AT25_PRED_PATH_IMAGES, GDTH_PATH, AT25_OUTFILE_IMAGES, 0, TYPE
            
        elif TYPE == 'Points':
            print("****** Doing for AtlasNet-25 Points*******")
            return AT25_PRED_PATH_POINTS, GDTH_PATH, AT25_OUTFILE_POINTS, 0, TYPE

    elif METHOD == 'AtlasNet':
        if TYPE == 'Images':
            print("****** Doing for AtlasNet-Sph Images*******")
            return AT_PRED_PATH_IMAGES, GDTH_PATH, AT_OUTFILE_IMAGES, 0, TYPE
        
        elif TYPE == 'Points':
            print("****** Doing for AtlasNet-Sph Points*******")
            return AT_PRED_PATH_POINTS, GDTH_PATH, AT_OUTFILE_POINTS, 0, TYPE

    elif METHOD == 'meshrcnn':
        print("****** Doing for MeshRCNN Images *******")
        return MRCNN_PRED_PATH_IMAGES, MRCNN_PRED_PATH_IMAGES, MRCNN_OUTFILE_IMAGES, -1, 'Images2'
        
    elif METHOD == 'pixel2mesh':
        print("****** Doing for Pixel2Mesh Images *******")
        return P2M_PRED_PATH_IMAGES, P2M_PRED_PATH_IMAGES, P2M_OUTFILE_IMAGES, -1, 'Images2'
        
    elif METHOD == 'nmf':
        if TYPE == 'Images':
            print("****** Doing for NMF Images*******")
            return NMF_PRED_PATH_IMAGES, GDTH_PATH, NMF_OUTFILE_IMAGES, -1, TYPE
            
        elif TYPE == 'Points':
            print("****** Doing for NMF Points*******")
            return NMF_PRED_PATH_POINTS, GDTH_PATH, NMF_OUTFILE_POINTS, -1, TYPE

    elif METHOD =='ablation':
            print("****** Doing for Ablation: {} *******".format(TYPE))
            return ABL_PRED_PATH_POINTS, GDTH_PATH, ABL_OUTFILE_POINTS, -1, 'Points'
            
    elif METHOD[:-2] == 'occnet':
        if TYPE == 'Images':
            print("****** Doing for OccNet-{} Images*******".format(format(METHOD[-1])))
            return OCNN_PRED_PATH_IMAGES, GDTH_PATH, OCNN_OUTFILENAME_IMAGES, 0, 'Images3'
        
        elif TYPE == 'Points':
            print("****** Doing for OccNet-{} Points*******".format(format(METHOD[-1])))
            return OCNN_PRED_PATH_POINTS, GDTH_PATH, OCNN_OUTFILE_POINTS, 0, TYPE
    
    return PRED_PATH, GDTH_PATH, OUTFILENAME, ROT, TYPE
