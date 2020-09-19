import torch
import numpy as np
import argparse
from tools import *
from torch.utils import data
import json
import os

class Dataset(data.Dataset):
    
    def __init__(self, METHOD, type, cat, PRED_PATH, GDTH_PATH, INITIAL, FINAL, ROT):
        
        self.PRED_PATH = PRED_PATH
        self.GDTH_PATH = GDTH_PATH
        self.cat = cat
        self.models = []
        self.type = type
        
        with open('cache/'+METHOD+'-'+cat+'.json', 'r') as file:        
            self.models = json.load(file)[INITIAL:FINAL]
            
        if ROT==-1:
            self.ROT=None
        else:
            self.ROT=ROT
        self.RESULTS = METRICS.copy()
        
    def __len__(self):
        return len(self.models)
    
    def __getitem__(self,index):
        
        model = self.models[index]
        if self.type == 'Images':
            pred_path = self.PRED_PATH + self.cat+'/'+model
            gdth_path = self.GDTH_PATH + self.cat+'/'+model[:-7]
        elif self.type == 'Points':
            pred_path = self.PRED_PATH + self.cat+'/'+model
            gdth_path = self.GDTH_PATH + self.cat+'/'+model.split('.')[0]
        elif self.type == 'Images2':
            pred_path = self.PRED_PATH + self.cat+'/'+model+ '/pred.obj'
            gdth_path = self.GDTH_PATH + self.cat+'/'+model+ '/gt.obj'
        elif self.type == 'Images3':
            pred_path = self.PRED_PATH + self.cat+'/'+model
            gdth_path = self.GDTH_PATH + self.cat+'/'+model[:-4]
            
        else:
            print("Invalid type supplied to dataloader")
            return
        
#         geo_score = compute_geometric_metrics_points(pred_path, gdth_path, rot=self.ROT)
#         manifold_score = calculate_manifoldness(pred_path)
        try:
            if self.type=='Images2':
                geo_score = compute_geometric_metrics(pred_path, gdth_path)
            else:
                geo_score = compute_geometric_metrics_points(pred_path, gdth_path, rot=self.ROT)
            
            manifold_score = calculate_manifoldness(pred_path)
        
        except:
            print("Something is wrong")
            return torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]), \
                torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]), \
                torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]), \
                torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]),\
                torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]), \
                torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]), \
                torch.FloatTensor([0]), torch.FloatTensor([0]), \
                torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0]), torch.FloatTensor([0])
        
        chamfer = geo_score['Chamfer-L2']
        normalconsistency = geo_score['NormalConsistency']
        absnormalconsistency = geo_score['AbsNormalConsistency']
        precision1 = geo_score['Precision@0.010000']
        recall1 = geo_score['Recall@0.010000']
        f11 = geo_score['F1@0.010000']
        
        precision2 = geo_score['Precision@0.030000']
        recall2 = geo_score['Recall@0.030000']
        f12 = geo_score['F1@0.030000']
        
        precision3 = geo_score['Precision@0.050000']
        recall3 = geo_score['Recall@0.050000']
        f13 = geo_score['F1@0.050000']
        
        precision4 = geo_score['Precision@0.070000']
        recall4 = geo_score['Recall@0.070000']
        f14 = geo_score['F1@0.070000']
        
        precision5 = geo_score['Precision@0.100000']
        recall5 = geo_score['Recall@0.100000']
        f15 = geo_score['F1@0.100000']
        
        vertices = torch.FloatTensor([manifold_score[0]])
        edges = torch.FloatTensor([manifold_score[1]])
        faces = torch.FloatTensor([manifold_score[2]])
        nmvertices = torch.FloatTensor([manifold_score[3]])
        nmedges = torch.FloatTensor([manifold_score[4]])
        nmfaces = torch.FloatTensor([manifold_score[5]])
        intersections = torch.FloatTensor([manifold_score[6]])
        
        nmvertices_ratio = torch.FloatTensor([manifold_score[3]/manifold_score[0]])
        nmedges_ratio = torch.FloatTensor([manifold_score[4]/manifold_score[1]])
        nmfaces_ratio = torch.FloatTensor([manifold_score[5]/manifold_score[1]])
        intersection_ratio = torch.FloatTensor([manifold_score[6]/manifold_score[2]])
        
        return chamfer, normalconsistency, absnormalconsistency, precision1, \
                recall1, f11, precision2, recall2, f12, precision3, recall3, f13, \
                precision4, recall4, f14, precision5, recall5, f15, vertices, edges, \
                faces, nmvertices, nmedges, nmfaces, intersections, nmvertices_ratio, \
                nmedges_ratio, nmfaces_ratio, intersection_ratio, torch.FloatTensor([1])
    
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, help = "method")
    parser.add_argument("--cat", type=str, help = "Category to evaluate")
    parser.add_argument("--init", type=int, help = "starting model")
    parser.add_argument("--final", type=int, help = "final model")
    parser.add_argument("--predpath", type=str, help = "predpath")
    parser.add_argument("--gdthpath", type=str, help = "gdthpath")
    parser.add_argument("--outfile", type=str, help = "outfile")
    parser.add_argument("--rot", type=int, help = "rotation")
    parser.add_argument("--type", type=str, help = "rotation")

    opt = parser.parse_args()

    METHOD = opt.method
    CATEGORY = opt.cat
    INITIAL = opt.init
    FINAL = opt.final
    PRED_PATH = opt.predpath
    GDTH_PATH = opt.gdthpath
    OUTFILENAME = opt.outfile
    TYPE = opt.type

    if opt.rot == 0:
        ROT = ['y',-90]
    elif opt.rot == 1:
        ROT = ['y',90]
    else:
        ROT=-1

    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}

    mydataset = Dataset(METHOD, TYPE, CATEGORY, PRED_PATH, GDTH_PATH, INITIAL, FINAL, ROT)
    generator = data.DataLoader(mydataset, **params)

    with open(OUTFILENAME, 'r') as file:
        RESULTS = json.load(file)

    batch_counter = INITIAL
    for model_metrics in generator:
        chamfer, normalconsistency, absnormalconsistency, precision1, \
        recall1, f11, precision2, recall2, f12, precision3, recall3, f13, \
        precision4, recall4, f14, precision5, recall5, f15, vertices, edges, \
        faces, nmvertices, nmedges, nmfaces, intersections, nmvertices_ratio, \
        nmedges_ratio, nmfaces_ratio, intersection_ratio, count = model_metrics

        RESULTS[CATEGORY]['Chamfer-L2'] += chamfer.sum().item()
        RESULTS[CATEGORY]['NormalConsistency'] += normalconsistency.sum().item()
        RESULTS[CATEGORY]['AbsNormalConsistency'] += absnormalconsistency.sum().item()
        RESULTS[CATEGORY]['Precision@0.01'] += precision1.sum().item()
        RESULTS[CATEGORY]['Recall@0.01'] += recall1.sum().sum().item()
        RESULTS[CATEGORY]['F1@0.01'] += f11.sum().item()
        RESULTS[CATEGORY]['Precision@0.03'] += precision2.sum().item()
        RESULTS[CATEGORY]['Recall@0.03'] += recall2.sum().item()
        RESULTS[CATEGORY]['F1@0.03'] += f12.sum().item()
        RESULTS[CATEGORY]['Precision@0.05'] += precision3.sum().item()
        RESULTS[CATEGORY]['Recall@0.05'] += recall3.sum().item()
        RESULTS[CATEGORY]['F1@0.05'] += f13.sum().item()
        RESULTS[CATEGORY]['Precision@0.07'] += precision4.sum().item()
        RESULTS[CATEGORY]['Recall@0.07'] += recall4.sum().item()
        RESULTS[CATEGORY]['F1@0.07'] += f14.sum().item()
        RESULTS[CATEGORY]['Precision@0.1'] += precision5.sum().item()
        RESULTS[CATEGORY]['Recall@0.1'] += recall5.sum().item()
        RESULTS[CATEGORY]['F1@0.1'] += f15.sum().item()
        RESULTS[CATEGORY]['Vertices'] += vertices.sum().item()
        RESULTS[CATEGORY]['Edges'] += edges.sum().item()
        RESULTS[CATEGORY]['Faces'] += faces.sum().item()
        RESULTS[CATEGORY]['NMVertices'] += nmvertices.sum().item()
        RESULTS[CATEGORY]['NMEdges'] += nmedges.sum().item()
        RESULTS[CATEGORY]['NMFaces'] += nmfaces.sum().item()
        RESULTS[CATEGORY]['Intersections'] += intersections.sum().item()
        RESULTS[CATEGORY]['NMVertices_ratio'] += nmvertices_ratio.sum().item()
        RESULTS[CATEGORY]['NMEdges_ratio'] += nmedges_ratio.sum().item()
        RESULTS[CATEGORY]['NMFaces_ratio'] += nmfaces_ratio.sum().item()
        RESULTS[CATEGORY]['Intersections_ratio'] += intersection_ratio.sum().item()
        RESULTS[CATEGORY]['Instances'] += count.sum().item()

#         batch_counter+=params['batch_size']
#         print("Progress: {}".format(batch_counter))
#         if batch_counter>4:
#             break

    with open(OUTFILENAME, 'w') as file:
        json.dump(RESULTS.copy(), file)

if __name__ == '__main__':
    main()