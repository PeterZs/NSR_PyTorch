import os
import json
import numpy as np
from subprocess import call, DEVNULL
import argparse
import pandas as pd
from utils import sphere_hammersley_sequence

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
from fnmatch import fnmatch
def _render(file_path, output_dir, num_views):
    yaws = []
    pitchs = []

    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    print(views)
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join( 'dataset_toolkits/blender_script', 'render.py'), 
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_dir,
        '--engine', 'CYCLES',
        '--save_mesh',
    ]

    call(args, stdout=DEVNULL, stderr=DEVNULL)

dataset_dir = 'dataset/ShapeNetCore'
filter_file = 'test/02691156/7182efccab0d3553c27f2d9f006d69eb.txt'
cate_list = ['02691156']

num_views = 5

print('Checking blender...', flush=True)
_install_blender()

for category in cate_list:
    print(category)
    obj_root_file =os.path.join(dataset_dir, category,  category)
    filter_arr = np.loadtxt(filter_file, dtype=str)

    render_obj_list = []
    for obj_id in filter_arr:
        if os.path.exists(os.path.join(obj_root_file,obj_id, 'models', 'renders')):
            continue
        render_obj_list.append((os.path.join(obj_root_file,obj_id, 'models'), 'model_normalized.obj'))

    for idx, (path, name) in enumerate(render_obj_list):
        os.makedirs(os.path.join(path, 'renders'), exist_ok=True)
        print('Rendering object...',str(idx),str(len(render_obj_list)), flush=True)
        print(os.path.join(path,name))
        _render(os.path.join(path,name), os.path.join(path, 'renders') , num_views)
        print('Render complete.')