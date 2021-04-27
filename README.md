# Contrastive Spatial Reasoning on Multi-View Line Drawings

[**Siyuan Xiang**](), [**Anbang Yang**](https://github.com/endeleze), [**Yanfei Xue**](),[**Yaoqing Yang**](), [**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng)

## Abstract
Spatial reasoning on multi-view line drawings by state-of-the-art supervised deep networks is recently shown with puzzling low performances on the SPARE3D dataset. To study the reason behind the low performance and to further our understandings of these geometric visual reasoning tasks, we design controlled experiments on both input data and baseline networks.
Guided by the hindsight from these experiment results, we propose a simple contrastive learning approach along with other network modifications to improve the baseline performance. 
Our approach uses a self-supervised binary classification network to compare the line drawing differences between various views of any two similar 3D objects.
It enables deep networks to effectively learn detail-sensitive yet view-invariant line drawing representations of 3D objects. 
Experiments show that our method could significantly increase the baseline performance in SPARE3D, while some other popular self-supervised learning methods cannot.
## Dependencies
Requires Python3.x, PyTorch, PythonOCC. Running on GPU is highly recommended. The code has been tested on Python 3.8.5, PyTorch 1.8.0, with CUDA 11.1.

### Task code
We significantly improve the SPARE3D task performance. Specifically, we design a contrastive spatial reasoning method for the T2I task.
Code of three tasks(T2I, I2P, P2I) can be found under [Tasks folder](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks).
#### [Isometric to pose](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Isometric_to_pose)
Run ```I2P_trainer.py``` with the parameters explained in args in the code. Our exploration experiments are under [structure_explore folder](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Isometric_to_pose/Structure_explore). ```network_structure``` contains the controlled experiments on network structure, such as w/o adaptive pooling layer, dropout layer, fully connected layer, and whether use ImageNet pre-trained parameters. ```network_capacity``` contains the controlled experiments on width and depth of the baseline network.
#### [Pose to isometric](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Pose_to_isometric)
Run ```P2I_trainer.py```with the parameters explained in args in the code.
#### [Three_view_to_isometric](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Three_view_to_isometric)
[Ours](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Three_view_to_isometric/Ours) folder contains the contrastive spatial reasoning method. Run ```Three2I_trainer.py``` with parameters under [Contrastive_learning folder](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Three_view_to_isometric/Ours/Contrastive_learning). To fine tune the network, run```Three2I_opt2_trainer.py``` with parameters under [Fine_tune folder](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Three_view_to_isometric/Ours/Fine_tune).
### Data generation
Code of data generation can be found under [Data generation folder](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Data_generation)


#### [Attention map](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Data_generation/Attention_Map)
Generate attention maps of the trained model using ```attention_map.py``` with image path and the root of trained model path.
## [Paper (arXiv)](https://arxiv.org/abs/2103.16732)
To cite our paper:
```

```

## Acknowledgment
The research is supported by NSF Future Manufacturing program under EEC-2036870. Siyuan Xiang gratefully thanks the IDC Foundation for its scholarship. We also thank the anonymous reviewers for constructive feedback.
