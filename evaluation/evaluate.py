import os
import numpy as np
import json
from tools import METRICS, CLASS_NAMES
from config import get_configs
import multiprocessing
import time
import argparse

def worker(BATCH, METHOD, NUM, cat, START, END, PRED_PATH, GDTH_PATH, OUTFILENAME, ROT, TYPE):
    for i in range(START, END, BATCH):
        print("[Worker {}] Progress: {}%".format(NUM,(i-START)/((END-START))*100))
        os.system('python worker.py --method {} --cat {} --init {} \
                   --final {} --predpath {} --gdthpath {} --outfile {} --rot {} --type {}'.
                   format(METHOD, cat, i, i+BATCH, PRED_PATH, GDTH_PATH, OUTFILENAME, ROT, TYPE))
        
        
        
# METHOD = 'occnet-3'         # method to be evaluated
# TYPE = 'Images'             # SVR or AE task type
# BATCH = 10                  # Batch size used for evaluation
# NUM_WORKERS = 13            # Number of worker threads

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help = "method")
    parser.add_argument("--type", type=str, help = "Type: Images (SVR) or Point (AE)")
    parser.add_argument("--batch_size", default=10, type=int, help = "batch size used for evaluation")
    parser.add_argument("--num_workers",default=13, type=int, help = "number of threads to launch")
    
    opt = parser.parse_args()
    
    os.makedirs('./results/', exist_ok=True)
    os.makedirs('./cache/', exist_ok=True)
    
    PRED_PATH, GDTH_PATH, OUTFILENAME, ROT, TYPE = get_configs(opt.method, opt.type)
    
    # Create the outfile
    RESULTS = {}
    for cat in os.listdir(GDTH_PATH):
        RESULTS[cat] = METRICS.copy()
    
    with open(OUTFILENAME, 'w') as file:
        json.dump(RESULTS, file)
        
        
    # Load all the meshes to be evaluated
    for cat in os.listdir(GDTH_PATH):   # For each category
        print("Doing for : ", CLASS_NAMES[cat])
        all_models = []
        if TYPE == 'Images2':          # The directory structure is different for Pixel2Mesh and MeshRCNN so accomodate that
            all_models = os.listdir(PRED_PATH+cat)
        else:
            for model in os.listdir(PRED_PATH+cat):
                if model[-4:] == '.obj':
                    all_models.append(model)
                                                         
        # Create cache for each category
        with open('cache/'+opt.method+'-'+cat+'.json', 'w') as file:
            json.dump(all_models, file)

        p_threads = []
        BLOCK = int(len(all_models)/opt.num_workers)
        count=0
        for i in range(0, len(all_models), BLOCK):
            outfilename = OUTFILENAME[:-5]+'-'+str(count)+'.json'
            
            RESULTS = {}
            for cat2 in os.listdir(GDTH_PATH):
                RESULTS[cat2] = METRICS.copy()
    
            with open(outfilename, 'w') as file:
                json.dump(RESULTS, file)
                
            p = multiprocessing.Process(target=worker, args=(opt.batch_size, opt.method, count, cat, i, i+BLOCK, PRED_PATH, GDTH_PATH, outfilename, ROT, TYPE))
            p.start()
            p_threads.append(p)
            count+=1
        print("Total threads :", len(p_threads))
        print("Total models :", len(all_models))
        
        for p in p_threads:
            p.join()
        
        with open(OUTFILENAME) as file:
            results = json.load(file)
            
        for i in range(count):
            with open(OUTFILENAME[:-5]+'-'+str(0)+'.json') as file:
                temp = json.load(file)
                for key in list(temp[cat].keys()):
                    results[cat][key] += temp[cat][key]
                    
        with open(OUTFILENAME, 'w') as file:
            json.dump(results, file)