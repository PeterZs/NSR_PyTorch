import numpy as np
import os
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import open3d as o3d
from fnmatch import fnmatch
from tqdm import tqdm
import multiprocessing as mp
import json
import utils3d


def align_point_clouds(a, b, voxel_size=None):
    if voxel_size is not None:
        b_pcd = o3d.geometry.PointCloud()
        b_pcd.points = o3d.utility.Vector3dVector(b)
        b_pcd = b_pcd.voxel_down_sample(voxel_size=voxel_size)
        b = np.asarray(b_pcd.points)

        a_pcd = o3d.geometry.PointCloud()
        a_pcd.points = o3d.utility.Vector3dVector(a)
        a_pcd = a_pcd.voxel_down_sample(voxel_size=voxel_size)
        a = np.asarray(a_pcd.points)
    kd_tree = cKDTree(b)
    def distance_sum(translation):
        tx, ty = translation
        translated_a = a + np.array([tx, ty, 0])
        distances, _ = kd_tree.query(translated_a)
        return np.sum(distances)

    initial_translation = np.array([0.0, 0.0])
    result = minimize(distance_sum, initial_translation, method='L-BFGS-B')

    optimal_translation = result.x
    trans = np.array([optimal_translation[0], optimal_translation[1], 0.0])
    return trans

def get_vertex_from_obj(mesh_file):
    if not os.path.exists(mesh_file):
        voxelize(os.path.dirname(mesh_file))
    vertices = utils3d.io.read_ply(mesh_file)[0]
    vertices_tr = np.vstack((-vertices[:, 1], vertices[:, 2], -vertices[:, 0])).T
    return vertices, vertices_tr


def parse_txt_points(shape_name):
    file = open(shape_name, 'r')
    lines = file.readlines()
    file.close()
    points = []
    labels = []
    for i in range(len(lines)):
        line = lines[i].split()
        points.append([-float(line[0]),float(line[1]),float(line[2])])
        labels.append(int(float(line[6])))
    point_num = len(labels)
    return points, labels, point_num

def voxelize(file):
    model_mesh_path = os.path.join(file, 'renders/mesh.ply')
    mesh = o3d.io.read_triangle_mesh(model_mesh_path)
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    utils3d.io.write_ply(os.path.join(file, 'voxelize.ply'), vertices)

def normalise_points(points):
    points = np.array(points)
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    scale = 1 / np.max(bbox_max - bbox_min)
    offset = -(bbox_min + bbox_max) / 2
    center_offset = (bbox_min + bbox_max) / 2  # Get the center of the bounding box
    normalized_points = (points - center_offset) * scale 
    return normalized_points

def allign(args):
    obj_class = part_class.split('_')[0]
    points, labels, point_num = parse_txt_points(os.path.join(part_info_path, part_class, 'points', args[1]))
    points_norm = normalise_points(points)
    vertex, vertex_tr = get_vertex_from_obj(os.path.join(obj_path, obj_class, obj_class,  args[1][:-4], 'models','voxelize.ply'))
    points = normalise_points(points)
    allign_metrix = align_point_clouds(points_norm.tolist(), vertex_tr.tolist(), 64)
    data = {
        'allign' : allign_metrix.tolist()
    }
    with open(os.path.join(allign_output_file, args[1][:-4]+'.json'), "w") as f:
        json.dump(data, f, indent=4) 
    print('process finish')
    return True



def get_vertices_labels(vertex, points, labels):
    tree = cKDTree(points)
    _, indices = tree.query(vertex)
    a_labels = [labels[i] for i in indices]
    return a_labels

def parse_txt_points(shape_name):
    file = open(shape_name, 'r')
    lines = file.readlines()
    file.close()
    points = []
    labels = []
    for i in range(len(lines)):
        line = lines[i].split()
        points.append([-float(line[0]),float(line[1]),float(line[2])])
        labels.append(int(float(line[6])))
    point_num = len(labels)
    return points, labels, point_num

def caculate_label(args):
    points_file = args[1]
    points, labels, point_num = parse_txt_points(os.path.join(part_info_path, part_class, 'points', points_file))
    labels_num = len(np.unique(np.array(labels)))
    with open(os.path.join(part_info_path, part_class, 'allign', points_file[:-4]+'.json'), "r") as f:
        loaded_data = json.load(f)
    
    points_norm = normalise_points(points)
    points = np.array(points_norm + np.array(loaded_data.get('allign')))
    
    mesh_path = os.path.join(obj_path, obj_class, obj_class,  args[1][:-4], 'models', 'voxelize.ply')
    
    vertices, vertices_tr = get_vertex_from_obj(mesh_path)
    vertex_labels = get_vertices_labels(vertices_tr, points, labels)

    label_json = {
        "labels":vertex_labels
    }
    with open(os.path.join(gt_output_file, obj_class, args[1][:-4]+'.json'), "w") as f:
        json.dump(label_json, f, indent=4) 
    print('process finish')
    return vertex_labels


obj_path = 'dataset/ShapeNetCore'
part_info_path = 'dataset/PartNet'

part_class = '02691156_airplane'
obj_class = part_class.split('_')[0]
gt_output_file = 'dataset/PartGT'


allign_output_file = os.path.join(part_info_path, part_class, 'allign')

pattern = "*.txt"
filter_file = 'test/02691156/7182efccab0d3553c27f2d9f006d69eb.txt'
filter_arr = np.loadtxt(filter_file, dtype=str)
args = []
for path, subdirs, files in os.walk(os.path.join(part_info_path, part_class, 'points')):
    for name in files:
        if fnmatch(name, pattern):
            if name.split('.')[0] in filter_arr:
                args.append((path, name))
os.makedirs(allign_output_file, exist_ok=True)
print(f"len of the arrays {len(args)}")

print(len(args))
workers = 40
pool = mp.Pool(workers)

with tqdm(total=len(args)) as pbar:
    for _ in pool.imap_unordered(allign, args):
        pbar.update(1)

pool.close()
pool.join()


args = []
for path, subdirs, files in os.walk(os.path.join(part_info_path, part_class, 'points')):
    for name in files:
        if fnmatch(name, pattern):
            if name.split('.')[0] in filter_arr:
                args.append((path, name))

os.makedirs(os.path.join(gt_output_file, obj_class), exist_ok=True)
print(f"len of the arrays {len(args)}")

workers = 40
pool = mp.Pool(workers)

with tqdm(total=len(args)) as pbar:
    for _ in pool.imap_unordered(caculate_label, args):
        pbar.update(1)
pool.close()
pool.join()