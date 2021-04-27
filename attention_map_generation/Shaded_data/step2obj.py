import sys
sys.path.append('D:\FreeCAD/bin')
import FreeCAD

import math
import Part
import Mesh
import os

stp = "./Line_Drawing"
obj = "./Shaded_obj"
if not os.path.exists(obj):
    os.mkdir(obj)

dics=sorted(os.listdir(stp))
list=['/0','/1','/2','/3']
for dic in dics[23:]:
    path=stp+'/'+dic
    des=obj+'/'+dic
    if not os.path.exists(des):
        os.mkdir(des)
    for dir in list:
        file=path+dir+'.step'
        shape = Part.Shape()
        shape.read(file)
        mesh = Mesh.Mesh()
        mesh.addFacets(shape.tessellate(0.01))
        path1=des+dir+'.obj'
        mesh.write(path1)