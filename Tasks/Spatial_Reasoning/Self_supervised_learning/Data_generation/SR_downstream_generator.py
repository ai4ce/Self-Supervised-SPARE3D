import shutil
from model2svg import *
from boolean import *

from gevent import Timeout

parser = argparse.ArgumentParser()
parser.add_argument('-pathread', type=str, help='file or folder to be processed.')
parser.add_argument('-pathwrite', type=str, help='file or folder to write.')
parser.add_argument('--reference', type=str,default='.', help='file or folder to write.')
parser.add_argument('-v', '--vps', type=str, default='ftr1234567891011121314151617181920', help='viewpoint(s) per file.')
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

train_fnames,valid_fnames=[],[]
train_src=os.path.join(pathread,'train')
valid_src=os.path.join(pathread,'valid')
if not os.path.exists(pathwrite):
    os.makedirs(os.path.join(pathwrite,'train'))
    os.mkdir(os.path.join(pathwrite, 'valid'))
    for v in os.listdir(train_src):
        train_fnames.append(os.path.join(train_src, v))
    for v in os.listdir(valid_src):
        valid_fnames.append(os.path.join(valid_src, v))
else:
    train_diro=sorted(os.listdir(os.path.join(pathwrite,'train')))
    valid_diro = sorted(os.listdir(os.path.join(pathwrite, 'valid')))
    for v in train_diro:
        if len(os.listdir(os.path.join(pathwrite,'train',v[:8])))<23:
            shutil.rmtree(os.path.join(pathwrite,'train', v[:8]))
            train_diro.remove(v)
    for v in os.listdir(train_src):
        if v[:8] not in train_diro:
            train_fnames.append(os.path.join(train_src, v))

    for v in valid_diro:
        if len(os.listdir(os.path.join(pathwrite, 'train', v[:8]))) < 23:
            shutil.rmtree(os.path.join(pathwrite, 'train', v[:8]))
            valid_diro.remove(v)
    for v in os.listdir(valid_src):
        if v[:8] not in valid_diro:
            valid_fnames.append(os.path.join(valid_src, v))

def Generate_task(fname, Path_output=pathwrite, args=args):
    converter = Model2SVG(width=args.width, height=args.height, tol=args.tol,
                          margin_left=args.margin_left, margin_top=args.margin_top,
                          line_width=args.line_width, line_width_hidden=args.line_width_hidden)
    index_list = ['1','9','2','11','3','13','4','15','5','10','6','12','7','14','8','16','17','18','19','20']
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
f = partial(Generate_task,Path_output=os.path.join(pathwrite,'train'), args=args)
t0 = time.time()
mask = p.map(f, train_fnames)
duration = time.time() - t0
p.close()
n_success = sum(mask)
print('{} done,  {} failed, elapsed time = {}!'.format(n_success, len(mask) - n_success,
                                                       time.strftime("%H:%M:%S", time.gmtime(duration))))

p = pool.Pool(processes=args.n_cores)
f = partial(Generate_task,Path_output=os.path.join(pathwrite,'valid'), args=args)
t0 = time.time()
mask = p.map(f, valid_fnames)
duration = time.time() - t0
p.close()
n_success = sum(mask)
print('{} done,  {} failed, elapsed time = {}!'.format(n_success, len(mask) - n_success,
                                                       time.strftime("%H:%M:%S", time.gmtime(duration))))

