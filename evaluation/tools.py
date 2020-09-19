import numpy as np
import trimesh
import open3d
import torch
import torch.nn.functional as F
import os
import json
from mesh_intersection.bvh_search_tree import BVH
'''
Note: If failed to load mesh_intersection. Do the following: 

cd ~/torch-mesh-isect
python3 setup.py install

Now you should be able to run this!
'''
from pytorch3d.ops import knn_gather, knn_points, sample_points_from_meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation as R

CLASS_NAMES = {
    "02828884": "bench",
    "03001627": "chair",
    "03636649": "lamp",
    "03691459": "speaker",
    "04090263": "firearm",
    "04379243": "table",
    "04530566": "watercraft",
    "02691156": "plane",
    "02933112": "cabinet",
    "02958343": "car",
    "03211117": "monitor",
    "04256520": "couch",
    "04401088": "cellphone",
    '02818832':'bed',
    '02843684':'birdcage',
    '02876657':'bottle',
    '02942699':'camera',
    '03797390':'cup',
    '02834778':'cycle',
    '02773838':'handbag',
    '02954340':'hat',
    '03261776':'headphone',
    '03513137':'helmet',
    '03085013':'keyboard',
    '03928116':'piano',
    '03938244':'pillow',
    '04004475':'printer',
    '02808440':'sink',
    '04460130':'tower',
    '02747177':'trashcan',
    '03046257':'clock'
}

METRICS = {'Chamfer-L2':0, 
           'NormalConsistency':0, 
           'AbsNormalConsistency':0,
           'Precision@0.01':0, 
           'Recall@0.01':0,
           'F1@0.01':0,
           'Precision@0.03':0, 
           'Recall@0.03':0,
           'F1@0.03':0,
           'Precision@0.05':0, 
           'Recall@0.05':0,
           'F1@0.05':0,
           'Precision@0.07':0, 
           'Recall@0.07':0,
           'F1@0.07':0,
           'Precision@0.1':0, 
           'Recall@0.1':0,
           'F1@0.1':0,
           'Vertices':0,
           'Edges':0,
           'Faces':0,
           'NMVertices':0,
           'NMEdges':0,
           'NMFaces':0,
           'Intersections':0,
           'NMVertices_ratio':0,
           'NMEdges_ratio':0,
           'NMFaces_ratio':0,
           'Intersections_ratio':0,
           'Instances': 0
          }

METRICSMANIFOLD = {'NMVertices_ratio':0,
                   'NMEdges_ratio':0,
                   'NMFaces_ratio':0,
                   'Intersections_ratio':0,
                   'Instances': 0
                  }


def calculate_manifoldness(mesh_path):
    '''
    This function evaluates a mesh based on the following metrics:
    manifold-edge (using open3D)
    manifold-vertex (using open3D)
    manifold-face  (adjacent face normal calculation). This is also used to compute normal consistecy sometimes
    mesh-intersections (using the torch-mesh-isect library)
    
    mesh_path: path to the mesh
    nv: number of vertices
    ne: number of edges
    nf: number of faces
    nm_edges: number of instances of non-manifold edges
    nm_vertices: number of instances of non-manifold vertices
    nm_faces: number of instances of non-manifold faces
    mesh_isect: number of instances of self-intersections (only 1 out of the two triangles is counted)
    '''
    
    nm_vertices, nm_edges = calculate_non_manifold_edge_vertex(mesh_path)
    nv, ne, nf, nm_faces, mesh_isect = calculate_non_manifold_face_intersection(mesh_path)
    
    return nv, ne, nf, nm_vertices, nm_edges, nm_faces, mesh_isect

def calculate_non_manifold_edge_vertex(mesh_path):
    '''
    This function returns the scores for non-manifold edge and vertices
    mesh_path: path to the .obj mesh object
    nm_edges: number of instances of non-manifold edges
    nm_vertices: number of instances of non-manifold vertices
    '''
    mesh = open3d.io.read_triangle_mesh(mesh_path)
    nm_edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    nm_vertices = np.asarray(mesh.get_non_manifold_vertices())
    return nm_vertices.shape[0], nm_edges.shape[0]

def calculate_non_manifold_face_intersection(mesh_path):
    '''
    This function returns the scores for non-manifold faces and amount of self-intersection
    mesh_path: path to the .obj mesh object
    nv: number of vertices
    ne: number of edges
    nf: number of faces
    nm_faces: number of instances of non-manifold faces
    mesh_isect: number of instances of self-intersections (only 1 out of the two triangles is counted)
    '''
    mesh = trimesh.load(mesh_path)
    f_adj = mesh.face_adjacency
    faces = mesh.faces
    fn = mesh.face_normals

    count=0
    for f in range(f_adj.shape[0]):
        if fn[f_adj[f,0]]@fn[f_adj[f,1]] < 0:
            count+=1
        
    vertices = torch.tensor(mesh.vertices,
                            dtype=torch.float32, device='cuda')
    faces = torch.tensor(mesh.faces.astype(np.int64),
                         dtype=torch.long,
                         device='cuda')

    batch_size = 1
    triangles = vertices[faces].unsqueeze(dim=0).contiguous()

    m = BVH(max_collisions=8)

    outputs = m(triangles)
    outputs = outputs.detach().cpu().numpy().squeeze()

    collisions = outputs[outputs[:, 0] >= 0, :]
    
    return mesh.vertices.shape[0], mesh.edges.shape[0], mesh.faces.shape[0], count, collisions.shape[0]


def merge_meshes(mesh1, mesh2):
    '''
    This function merges two trimesh meshes into a single trimesh mesh. Note that it currectly does not support face colors.
    '''
    v1 = mesh1.vertices
    v2 = mesh2.vertices
    
    f1 = mesh1.faces
    f2 = mesh2.faces + v1.shape[0]
    
    v = np.vstack((v1,v2))
    f = np.vstack((f1, f2))
    
    return trimesh.Trimesh(vertices=v, faces=f)

def highlight_self_intersections(mesh_path):
    '''
    This function is used to create .obj file where self-intersections are highlighted
    mesh_path: path to the .obj object
    '''
    mesh = trimesh.load(mesh_path)
    vertices = torch.tensor(mesh.vertices,
                            dtype=torch.float32, device='cuda')
    faces = torch.tensor(mesh.faces.astype(np.int64),
                         dtype=torch.long,
                         device='cuda')

    batch_size = 1
    triangles = vertices[faces].unsqueeze(dim=0)

    m = BVH(max_collisions=8)

    outputs = m(triangles)
    outputs = outputs.detach().cpu().numpy().squeeze()

    collisions = outputs[outputs[:, 0] >= 0, :]

    print('Number of collisions = ', collisions.shape[0])
    print('Percentage of collisions (%)',
          collisions.shape[0] / float(triangles.shape[1]) * 100)
    
    recv_faces = mesh.faces[collisions[:, 0]]
    intr_faces = mesh.faces[collisions[:, 1]]
    mesh1 = trimesh.Trimesh(mesh.vertices, recv_faces)
    mesh2 = trimesh.Trimesh(mesh.vertices, intr_faces)
    
    inter_mesh = merge_meshes(mesh1,mesh2)
    fci = np.ones((2*recv_faces.shape[0],3))*np.array([1,0,0])
    fcf = np.ones((mesh.faces.shape[0],3))
    mesh = merge_meshes(inter_mesh, mesh)
    final_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=np.vstack((fci,fcf)))
#     final_mesh.export(mesh_path[:-4]+'_intersection.obj');

    if  collisions.shape[0] > 0:
        print(mesh_path)
        final_mesh.export('intersection.obj');
    
def regular_on_sphere_points(num):
    '''
    - Regular placement involves chosing points such that there one point per d_area
    References:
    Deserno, 2004, How to generate equidistributed points on the surface of a sphere
    http://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    Code source : https://gist.github.com/dinob0t/9597525
    '''
    import math
    r=1
    points = []
    #Break out if zero points
    if num==0:
        return points
    a = 4.0 * math.pi*(r**2.0 / num)
    d = math.sqrt(a)
    m_theta = int(round(math.pi / d))
    d_theta = math.pi / m_theta
    d_phi = a / d_theta
    for m in range(0,m_theta):
        theta = math.pi * (m + 0.5) / m_theta
        m_phi = int(round(2.0 * math.pi * math.sin(theta) / d_phi))
        for n in range(0,m_phi):
            phi = 2.0 * math.pi * n / m_phi
            x = r * math.sin(theta) * math.cos(phi)
            y = r * math.sin(theta) * math.sin(phi)
            z = r * math.cos(theta)
            points.append([x,y,z])
    return np.array(points)

def compute_geometric_metrics(pred_mesh_path, gdth_mesh_path):
    
    with torch.no_grad():
        pred_mesh = trimesh.load(pred_mesh_path)
        gdth_mesh = trimesh.load(gdth_mesh_path)

        pred_vertices = pred_mesh.vertices
        gdth_vertices = gdth_mesh.vertices

        pred_pts = pred_mesh.sample(10000)
        gdth_pts = gdth_mesh.sample(10000)

        pred_vertices -= np.mean(pred_pts,axis=0)
        gdth_vertices -= np.mean(gdth_pts,axis=0)

        pred_vertices /= np.max(np.linalg.norm(pred_vertices, axis=1))
        gdth_vertices /= np.max(np.linalg.norm(gdth_vertices, axis=1))

    #     sph = regular_on_sphere_points(1024)
    #     both =  np.vstack((np.vstack((pred_vertices, gdth_vertices)), sph))
    #     np.savetxt('test.xyz',both)

        pred_vertices = torch.from_numpy(pred_vertices).float()
        gdth_vertices = torch.from_numpy(gdth_vertices).float()

        pred_faces = torch.from_numpy(pred_mesh.faces)
        gdth_faces = torch.from_numpy(gdth_mesh.faces)

        pred_mesh = Meshes(verts = [pred_vertices], faces = [pred_faces])
        gdth_mesh = Meshes(verts = [gdth_vertices], faces = [gdth_faces])

        pred_points, pred_normals = sample_points_from_meshes(pred_mesh,num_samples=10000,return_normals=True)
        gt_points, gt_normals = sample_points_from_meshes(gdth_mesh,num_samples=10000,return_normals=True)

        metrics = _compute_sampling_metrics(pred_points, pred_normals, gt_points, gt_normals, eps=1e-8)

    return metrics

def compute_geometric_metrics_points(pred_mesh_path, gdth_mesh_path, rot=None):
    
    with torch.no_grad():
        pred_mesh = trimesh.load(pred_mesh_path)
        gt_points = np.load(gdth_mesh_path+'/points.npy')
        gt_normals = np.load(gdth_mesh_path+'/normals.npy')
        mask = np.random.randint(0, gt_points.shape[0], 10000)
        gt_points = gt_points[mask,:]
        gt_normals = gt_normals[mask,:]
        
        if rot is not None:
            r = R.from_euler(rot[0], rot[1], degrees=True) 
            gt_points = r.apply(gt_points)
            gt_normals = r.apply(gt_normals)
            
            
        # make to unit scale and shift to origin
        gt_points -= np.mean(gt_points, axis=0)
        gt_points /= np.max(np.linalg.norm(gt_points, axis=1))

#         pred_vertices = pred_mesh.verts_list()[0]
        pred_vertices = pred_mesh.vertices
        pred_pts = pred_mesh.sample(10000)
        pred_vertices -= np.mean(pred_pts,axis=0)
        pred_vertices /= np.max(np.linalg.norm(pred_vertices, axis=1))
        
#         sph = regular_on_sphere_points(1024)
#         both =  np.vstack((np.vstack((pred_vertices, gt_points)), sph))
#         np.savetxt('test.xyz',both)
        
        pred_vertices = torch.from_numpy(pred_vertices).float()
        pred_faces = torch.from_numpy(pred_mesh.faces)
        pred_mesh = Meshes(verts = [pred_vertices], faces = [pred_faces])
        
        pred_points, pred_normals = sample_points_from_meshes(pred_mesh,num_samples=10000,return_normals=True)

        pred_points = pred_points
        pred_normals = pred_normals
        gt_points = torch.from_numpy(gt_points).type_as(pred_points).unsqueeze(0)
        gt_normals = torch.from_numpy(gt_normals).type_as(pred_points).unsqueeze(0)

        metrics = _compute_sampling_metrics(pred_points, pred_normals, gt_points, gt_normals, eps=1e-8)

    return metrics

def _compute_sampling_metrics(pred_points, pred_normals, gt_points, gt_normals, eps):
    """
    Compute metrics that are based on sampling points and normals:
    - L2 Chamfer distance
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds
    - Normal consistency (if normals are provided)
    - Absolute normal consistency (if normals are provided)
    Inputs:
        - pred_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each predicted mesh
        - pred_normals: Tensor of shape (N, S, 3) giving normals of points sampled
          from the predicted mesh, or None if such normals are not available
        - gt_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each ground-truth mesh
        - gt_normals: Tensor of shape (N, S, 3) giving normals of points sampled from
          the ground-truth verts, or None of such normals are not available
        - thresholds: Distance thresholds to use for precision / recall / F1
        - eps: epsilon value to handle numerically unstable F1 computation
    Returns:
        - metrics: A dictionary where keys are metric names and values are Tensors of
          shape (N,) giving the value of the metric for the batch
    """
    metrics = {}
    thresholds = [0.01, 0.03, 0.05, 0.07, 0.1]
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)
    if gt_normals is not None:
        pred_normals_near = knn_gather(gt_normals, knn_pred.idx, lengths_gt)[..., 0, :]  # (N, S, 3)
    else:
        pred_normals_near = None

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    if pred_normals is not None:
        gt_normals_near = knn_gather(pred_normals, knn_gt.idx, lengths_pred)[..., 0, :]  # (N, S, 3)
    else:
        gt_normals_near = None

    # Compute L2 chamfer distances
    chamfer_l2 = pred_to_gt_dists2.mean(dim=1) + gt_to_pred_dists2.mean(dim=1)
    metrics["Chamfer-L2"] = chamfer_l2

    # Compute normal consistency and absolute normal consistance only if
    # we actually got normals for both meshes
    if pred_normals is not None and gt_normals is not None:
        pred_to_gt_cos = F.cosine_similarity(pred_normals, pred_normals_near, dim=2)
        gt_to_pred_cos = F.cosine_similarity(gt_normals, gt_normals_near, dim=2)

        pred_to_gt_cos_sim = pred_to_gt_cos.mean(dim=1)
        pred_to_gt_abs_cos_sim = pred_to_gt_cos.abs().mean(dim=1)
        gt_to_pred_cos_sim = gt_to_pred_cos.mean(dim=1)
        gt_to_pred_abs_cos_sim = gt_to_pred_cos.abs().mean(dim=1)
        normal_dist = 0.5 * (pred_to_gt_cos_sim + gt_to_pred_cos_sim)
        abs_normal_dist = 0.5 * (pred_to_gt_abs_cos_sim + gt_to_pred_abs_cos_sim)
        metrics["NormalConsistency"] = normal_dist
        metrics["AbsNormalConsistency"] = abs_normal_dist

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics