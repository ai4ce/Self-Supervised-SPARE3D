import argparse
import json
import os
from multiprocessing import pool, cpu_count
os.environ["PYOPENGL_PLATFORM"] = "egl"
import shutil
import sys
import numpy as np
import trimesh
import pyrender
import cv2
from functools import partial


def save_png(in_path,out_path,type,v):


    fuze_trimesh = trimesh.load(in_path)
    Material={
        'gold':pyrender.MetallicRoughnessMaterial(alphaMode='OPAQUE', alphaCutoff=0,
                                             baseColorFactor=(0.2157, 0.6863, 0.8314, 1), doubleSided=False,
                                             metallicFactor=0.6, roughnessFactor=0.5),
        'gypsum':pyrender.MetallicRoughnessMaterial(alphaMode='OPAQUE', alphaCutoff=0,
                                             baseColorFactor=(0.7922, 0.7882, 0.7765, 1), doubleSided=False,
                                             metallicFactor=0, roughnessFactor=1),
        'diamond':pyrender.MetallicRoughnessMaterial(alphaMode='BLEND', alphaCutoff=1,
                                                baseColorFactor=(255/256, 242/256, 185/256, 0.5), doubleSided=False,
                                                metallicFactor=0.5, roughnessFactor=0.5),
        'silver':pyrender.MetallicRoughnessMaterial(alphaMode='OPAQUE', alphaCutoff=0,
                                                baseColorFactor=(173/256, 169/256, 168/256, 1), doubleSided=False,
                                                metallicFactor=0.6, roughnessFactor=0.5)
    }
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh, material=Material[type])
    camera = pyrender.OrthographicCamera(xmag=mesh.scale, ymag=mesh.scale, znear=0.5, zfar=1000)

    s = np.sqrt(2)/2
    dis=mesh.scale
    dis_=dis/np.sqrt(3)
    DIRS={
    'f': np.array([
           [1.0, 0,   0.0, 0.0],
           [0.0,  0.0, -1.0, -dis],
           [0.0,  1.0,   0,   0],
           [0.0,  0.0, 0.0, 1.0],
        ]),
    'r': np.array([
           [0.0, 0,   1.0, dis],
           [1.0,  0.0, 0.0, 0],
           [0.0,  1.0,   0.0,   0],
           [0.0,  0.0, 0.0, 1.0],
        ]),
    't':  np.array([
           [1.0, 0,   0.0, 0],
           [0.0,  1.0, 0.0, 0],
           [0.0,  0.0,   1.0,   dis],
           [0.0,  0.0, 0.0, 1.0],
        ]),
    '1': np.array([
           [s, 0.5,   -0.5, -dis_],
           [-s,  0.5, -0.5, -dis_],
           [0,  s,   s,   dis_],
           [0.0,  0.0, 0.0, 1.0],
        ]),
    '2': np.array([
           [s, -0.5,   0.5, dis_],
           [s,  0.5, -0.5, -dis_],
           [0,  s,   s,   dis_],
           [0.0,  0.0, 0.0, 1.0],
        ]),
    '3': np.array([
           [-s, -0.5,   0.5, dis_],
           [s,  -0.5, 0.5, dis_],
           [0,  s,   s,   dis_],
           [0.0,  0.0, 0.0, 1.0],
        ]),
    '4': np.array([
        [-s, 0.5, -0.5, -dis_],
        [-s, -0.5, 0.5, dis_],
        [0, s, s, dis_],
        [0.0, 0.0, 0.0, 1.0],
        ]),
    '5': np.array([
           [s, -0.5,   -0.5, -dis_],
           [-s,  -0.5, -0.5, -dis_],
           [0,  s,   -s,   -dis_],
           [0.0,  0.0, 0.0, 1.0],
        ]),
    '6': np.array([
           [s, 0.5,   0.5, dis_],
           [s,  -0.5, -0.5, -dis_],
           [0,  s,   -s,   -dis_],
           [0.0,  0.0, 0.0, 1.0],
        ]),
    '7': np.array([
           [-s, 0.5,   0.5, dis_],
           [s,  0.5, 0.5, dis_],
           [0,  s,   -s,   -dis_],
           [0.0,  0.0, 0.0, 1.0],
        ]),
    '8': np.array([
           [-s, -0.5,   -0.5, -dis_],
           [-s,  0.5, 0.5, dis_],
           [0,  s,   -s,   -dis_],
           [0.0,  0.0, 0.0, 1.0],
        ]),
    }

    light_dis=mesh.scale/3
    light1_pose= np.array([
           [s, 0.5,   -0.5, -light_dis],
           [-s,  0.5, -0.5, -light_dis],
           [0,  s,   s,   light_dis],
           [0.0,  0.0, 0.0, 1.0],
        ])
    light1 = pyrender.DirectionalLight(color=np.ones(3), intensity=1)

    light2_pose= np.array([
           [s, -0.5,   0.5, light_dis],
           [s,  0.5, -0.5, -light_dis],
           [0,  s,   s,   light_dis],
           [0.0,  0.0, 0.0, 1.0],
        ])
    light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=1)
    light3_pose= np.array([
           [-s, -0.5,   0.5, light_dis],
           [s,  -0.5, 0.5, light_dis],
           [0,  s,   s,   light_dis],
           [0.0,  0.0, 0.0, 1.0],
        ])
    light3 = pyrender.DirectionalLight(color=np.ones(3), intensity=1)
    light4_pose= np.array([
        [-s, 0.5, -0.5, -light_dis],
        [-s, -0.5, 0.5, light_dis],
        [0, s, s, light_dis],
        [0.0, 0.0, 0.0, 1.0],
        ])
    light4 = pyrender.DirectionalLight(color=np.ones(3), intensity=1)
    light5_pose= np.array([
           [s, -0.5,   -0.5, -light_dis],
           [-s,  -0.5, -0.5, -light_dis],
           [0,  s,   -s,   -light_dis],
           [0.0,  0.0, 0.0, 1.0],
        ])
    light5 = pyrender.DirectionalLight(color=np.ones(3), intensity=1)
    light6_pose= np.array([
           [s, 0.5,   0.5, light_dis],
           [s,  -0.5, -0.5, -light_dis],
           [0,  s,   -s,   -light_dis],
           [0.0,  0.0, 0.0, 1.0],
        ])
    light6 = pyrender.DirectionalLight(color=np.ones(3), intensity=1)
    light7_pose= np.array([
           [-s, 0.5,   0.5, light_dis],
           [s,  0.5, 0.5, light_dis],
           [0,  s,   -s,   -light_dis],
           [0.0,  0.0, 0.0, 1.0],
        ])
    light7 = pyrender.DirectionalLight(color=np.ones(3), intensity=1)
    light8_pose= np.array([
           [-s, -0.5,   -0.5, -light_dis],
           [-s,  0.5, 0.5, light_dis],
           [0,  s,   -s,   -light_dis],
           [0.0,  0.0, 0.0, 1.0],
        ])
    light8 = pyrender.DirectionalLight(color=np.ones(3), intensity=1)
    lightf_pose = np.array([
        [1.0, 0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -light_dis],
        [0.0, 1.0, 0, 0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    lightf = pyrender.DirectionalLight(color=np.ones(3), intensity=0)
    lightr_pose = np.array([
        [0.0, 0, 1.0, light_dis],
        [1.0, 0.0, 0.0, 0],
        [0.0, 1.0, 0.0, 0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    lightr = pyrender.DirectionalLight(color=np.ones(3), intensity=0)
    lightt_pose = np.array([
        [1.0, 0, 0.0, 0],
        [0.0, 1.0, 0.0, 0],
        [0.0, 0.0, 1.0, light_dis],
        [0.0, 0.0, 0.0, 1.0],
    ])
    lightt = pyrender.DirectionalLight(color=np.ones(3), intensity=0)
    lightre_pose = np.array([
        [1.0, 0, 0.0, 0.0],
        [0.0, 0.0, 1.0, light_dis],
        [0.0, -1.0, 0, 0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    lightre = pyrender.DirectionalLight(color=np.ones(3), intensity=0)
    lightl_pose = np.array([
        [0.0, 0, -1.0, -light_dis],
        [-1.0, 0.0, 0.0, 0],
        [0.0, 1.0, 0.0, 0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    lightl = pyrender.DirectionalLight(color=np.ones(3), intensity=0)
    lightb_pose = np.array([
        [-1.0, 0, 0.0, 0],
        [0.0, 1.0, 0.0, 0],
        [0.0, 0.0, -1.0, -light_dis],
        [0.0, 0.0, 0.0, 1.0],
    ])
    lightb = pyrender.DirectionalLight(color=np.ones(3), intensity=0)

    scene = pyrender.Scene()
    scene.add(mesh)

    scene.add(camera, pose=DIRS[v])
    scene.add(light1, pose=light1_pose)
    scene.add(light2, pose=light2_pose)
    scene.add(light3, pose=light3_pose)
    scene.add(light4, pose=light4_pose)
    scene.add(light5, pose=light5_pose)
    scene.add(light6, pose=light6_pose)
    scene.add(light7, pose=light7_pose)
    scene.add(light8, pose=light8_pose)
    r = pyrender.OffscreenRenderer(200, 200)
    color, depth = r.render(scene)
    r.delete()
    cv2.imwrite(out_path, color)

def Generate_task(fname,answer_dic,pose_dic,Path_output,type,args):
    path_list = fname.split(os.sep)
    model_number = path_list[-1].replace(".obj", "")

    MotherDic = Path_output + "/" + model_number + "/"

    if not os.path.exists(MotherDic):
        os.makedirs(MotherDic)

    viewpoints = ['f', 'r', 't']
    answer_number = ['0', '1', '2', '3']
    answer_number.remove(str(answer_dic[model_number]))

    try:
        for vp in viewpoints:
            save_png(fname,MotherDic + model_number + "_" + vp + ".png",type,vp)

        index = str(pose_dic[model_number][0])
        save_png(fname, MotherDic + str(answer_dic[model_number]) + ".png",type, index)

        f = open(MotherDic + "pose_" + index + ".txt", "w")
        f.close()
        Index_list=pose_dic[model_number][1]
        j = 0
        for Vp in Index_list:
            save_png(fname, MotherDic + answer_number[j] + ".png",type, str(Vp))
            j = j + 1
        return 1

    except Exception as re:
        shutil.rmtree(MotherDic)
        print(fname + ' failed, due to: {}'.format(re))
        return 0

def main(args):
    pathread = args.inf
    pathwrite = args.outf
    type=args.type
    if not os.path.exists(pathwrite):
        os.mkdir(pathwrite)
    fname_answer = os.path.join(pathread, 'answer.json')
    shutil.copy(fname_answer,pathwrite)
    with open(fname_answer, 'r') as ff:
        answer = json.load(ff)
    fname_pose = os.path.join(pathread, 'pose.json')
    with open(fname_pose, 'r') as ff:
        pose = json.load(ff)
    index = sorted(os.listdir(pathread))
    index.remove('answer.json')
    index.remove('pose.json')

    fnames = []

    for v in answer:
        cad_name = os.path.join(pathread, v)
        cad_name=cad_name+'.obj'
        fnames.append(cad_name)

    p = pool.Pool(processes=args.n_cores)
    f = partial(Generate_task,answer_dic=answer,pose_dic=pose,Path_output=pathwrite,type=type,args=args)
    results=p.map(f, fnames)
    print('{} files are successed, {} are failed'.format(sum(results),len(results)-sum(results)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-I','--inf',type=str,default='../obj_data', help='file or folder with obj files.')
    parser.add_argument('-O','--outf',type=str,default='../P2I_gold_test',help='output folder.')
    parser.add_argument('-T', '--type', type=str, default='gold', help='material type.')
    parser.add_argument('-n', '--n_cores', type=int, default=cpu_count(), help='number of processors.')
    args = parser.parse_args(sys.argv[1:])
    main(args)