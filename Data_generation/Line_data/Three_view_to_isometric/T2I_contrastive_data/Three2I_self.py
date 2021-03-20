import random
import os
import shutil
import argparse
from multiprocessing import pool, cpu_count
import glob
from model2svg import *
from boolean import *
import numpy as np
import json
from gevent import Timeout
from gevent import monkey

parser = argparse.ArgumentParser()
parser.add_argument('--pathread', default='../step_data', help='file or folder to be processed.')
parser.add_argument('--pathwrite', default='./self_data', help='file or folder to write.')
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

if not os.path.exists(pathwrite):
    os.makedirs(pathwrite)

dirs = sorted(os.listdir(pathread))
done_path='./self_data'
done=sorted(os.listdir(done_path))
for dic in done:
    dir=os.path.join(done_path,dic)
    if len(os.listdir(dir))!=8:
        shutil.rmtree(dir)
        done.remove(dic)
        print(dic)
fnames = []
for v in dirs:
    o=v.split(os.sep)[-1].replace('.step','')
    if o not in done:
        filename = os.path.join(pathread, v)
        #cad_name = glob.glob(filename + "/*.step")
        fnames.append(filename)


def Generate_task(fname, Path_output=pathwrite, args=args):
    converter = Model2SVG(width=args.width, height=args.height, tol=args.tol,
                          margin_left=args.margin_left, margin_top=args.margin_top,
                          line_width=args.line_width, line_width_hidden=args.line_width_hidden)

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

        #### generate F R T views
        simple_shp_1 = New_shp(boundbox[0], boundbox[1], boundbox[2], boundbox[3], boundbox[4], boundbox[5],
                               boundbox[6], boundbox[7], boundbox[8])
        shp_1 = BRepAlgoAPI_Cut(shp, simple_shp_1).Shape()

        simple_shp_2 = New_shp(boundbox[0], boundbox[1], boundbox[2], boundbox[3], boundbox[4], boundbox[5],
                               boundbox[6], boundbox[7], boundbox[8])
        shp_2 = BRepAlgoAPI_Cut(shp, simple_shp_2).Shape()
        for vp in viewpoints:
            converter.export_shape_to_svg(shape=shp_1, filename=MotherDic + model_number + "_" + vp+'left' + ".svg",
                                          proj_ax=converter.DIRS[vp], max_eadge=max_3d_eadge)
            #### generate correct answer
        converter.export_shape_to_svg(shape=shp_1, filename=MotherDic +'isometric_left' + ".svg",
                                      proj_ax=converter.DIRS["2"], max_eadge=max_3d_eadge)
        for vp in viewpoints:
            converter.export_shape_to_svg(shape=shp_2, filename=MotherDic + model_number + "_" + vp+'right' + ".svg",
                                          proj_ax=converter.DIRS[vp], max_eadge=max_3d_eadge)
            #### generate correct answer
        converter.export_shape_to_svg(shape=shp_2, filename=MotherDic +'isometric_right' + ".svg",
                                      proj_ax=converter.DIRS["2"], max_eadge=max_3d_eadge)

        ###  generate wrong answers

        return 1

    except Exception as re:
        shutil.rmtree(MotherDic)
        print(MotherDic + "has been removed")
        print(fname + ' failed, due to: {}'.format(re))
        return 0

p = pool.Pool(processes=args.n_cores)
f = partial(Generate_task)

t0 = time.time()
mask = p.map(f, fnames)
Mask = np.asarray(mask)

dirs_valid = np.delete(np.array(dirs), np.where(Mask == 0))


duration = time.time() - t0
p.close()
n_success = sum(mask)
print('{} done,  {} failed, elapsed time = {}!'.format(n_success, len(mask) - n_success,
                                                       time.strftime("%H:%M:%S", time.gmtime(duration))))