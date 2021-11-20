import random
import os
import shutil
import argparse
from multiprocessing import pool, cpu_count
import glob
from model2svg import *
import json
import numpy as np
from boolean import *

from gevent import Timeout
from gevent import monkey

parser = argparse.ArgumentParser()
parser.add_argument('-pathread', type=str, help='file or folder to be processed.')
parser.add_argument('-pathwrite', type=str, help='file or folder to write.')
parser.add_argument('--reference', type=str,default='.', help='file or folder to write.')
parser.add_argument('-v', '--vps', type=str, default='ftr12345678', help='viewpoint(s) per file.')
parser.add_argument('-n', '--n_cores', type=int, default=cpu_count(), help='number of processors.')
parser.add_argument('-W', '--width', type=int, default=200, help='svg width.')
parser.add_argument('-H', '--height', type=int, default=200, help='svg height.')
parser.add_argument('-t', '--tol', type=float, default=0.04, help='svg discretization tolerance.')
parser.add_argument('-ml', '--margin_left', type=int, default=1, help='svg left margin.')
parser.add_argument('-mt', '--margin_top', type=int, default=1, help='svg top margin.')
parser.add_argument('-lw', '--line_width', type=float, default=0.7, help='svg line width.')
parser.add_argument('-lwh', '--line_width_hidden', type=float, default=0.35, help='svg hidden line width.')
args = parser.parse_args()

pathread = args.pathread
pathwrite = args.pathwrite

dirs=[]
fnames=[]
print('Total data:{}'.format(len(dirs)))
with open(args.reference,'r') as f:
    pick=json.load(f)
valid=pick['60000']
for dir in valid:
    dirs.append(dir+'.step')
print('Total valid:{}'.format(len(dirs)))
if not os.path.exists(pathwrite):
    os.makedirs(pathwrite)
    for v in dirs:
        fnames.append(os.path.join(pathread, v))
else:
    diro = sorted(os.listdir(pathwrite))
    for v in diro:
        if len(os.listdir(os.path.join(pathwrite,v[:8])))<11:
            shutil.rmtree(os.path.join(pathwrite,v[:8]))
            diro.remove(v)
    for v in dirs:
        if v[:8] not in diro:
            fnames.append(os.path.join(pathread, v))
print('New to generate:{}'.format(len(fnames)))


def Generate_task(fname, Path_output=pathwrite, args=args):
    converter = Model2SVG(width=args.width, height=args.height, tol=args.tol,
                          margin_left=args.margin_left, margin_top=args.margin_top,
                          line_width=args.line_width, line_width_hidden=args.line_width_hidden)
    index_list = ['1', '2', '3', '4', '5', '6', '7', '8']
    path_list = fname.split(os.sep)
    model_number = path_list[-1].replace(".step", "")
    model_number = model_number[0: 8]

    MotherDic = Path_output + "/" + model_number + "/"

    if not os.path.exists(MotherDic):
        os.makedirs(MotherDic)

    viewpoints = ['f', 'r', 't']

    try:
        seconds = 60
        timeout = Timeout(seconds)
        timeout.start()
        shp = read_step_file(fname)
        boundbox = get_boundingbox(shp, use_mesh=False)
        max_3d_eadge = max(boundbox[6], boundbox[7], boundbox[8])
        # sc=min(args.width, args.height)/max_3d_eadge

        #### generate F R T views
        for vp in viewpoints:
            converter.export_shape_to_svg(shape=shp, filename=MotherDic + model_number + "_" + vp + ".svg",
                                          proj_ax=converter.DIRS[vp], max_eadge=max_3d_eadge)

        for Vp in index_list:
            converter.export_shape_to_svg(shape=shp, filename=MotherDic +  Vp+ ".svg",
                                          proj_ax=converter.DIRS[Vp], max_eadge=max_3d_eadge)
        return 1
    except Exception as re:
        shutil.rmtree(MotherDic)
        print(fname + ' failed, due to: {}'.format(re))
        return 0


p = pool.Pool(processes=args.n_cores)
f = partial(Generate_task, args=args)

t0 = time.time()
mask = p.map(f, fnames)

duration = time.time() - t0
p.close()
n_success = sum(mask)
print('{} done,  {} failed, elapsed time = {}!'.format(n_success, len(mask) - n_success,
                                                       time.strftime("%H:%M:%S", time.gmtime(duration))))

