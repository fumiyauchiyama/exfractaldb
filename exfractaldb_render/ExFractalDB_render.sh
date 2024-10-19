#!/bin/bash

#$-l rt_F=1
#$-l h_rt=24:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
module load python/3.10/3.10.14
pip3 install -r requirements.txt

export PYOPENGL_PLATFORM=egl
variance_threshold=0.05
numof_category=1000
param_path='../dataset/MVFractalDB-'${numof_category}'/3DIFS_params'
model_save_path='../dataset/MVFractalDB-'${numof_category}'/3Dmodels'
image_save_path='../dataset/MVFractalDB-'${numof_category}'/images'

# Parameter search
python3 3dfractal_render/category_search.py --variance=${variance_threshold} --numof_classes=${numof_category} --save_root=${param_path}

# Generate 3D fractal model
python3 3dfractal_render/instance.py --load_root ${param_path} --save_root ${model_save_path} --classes ${numof_category}

# Render Multi-view images
python3 image_render/render.py --load_root ${model_save_path} --save_root ${image_save_path}
