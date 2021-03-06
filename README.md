# Self-supervised Spatial Reasoning on Multi-View Line Drawings [CVPR 2022]

[Siyuan Xiang*](), [Anbang Yang*](https://github.com/endeleze), [Yanfei Xue](),[Yaoqing Yang](), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ)

**We significantly improve SPARE3D baselines using self-supervised learning approaches.**

<p align="center"><img src='docs/figs/teaser_fig_cut.JPG' align="center" height="500px"> </p>

Poster Page: https://ai4ce.github.io/Self-Supervised-SPARE3D/

[**ArXiv: Self-supervised Spatial Reasoning on Multi-View Line Drawings**](https://arxiv.org/abs/2104.13433)     

## Abstract
Spatial reasoning on multi-view line drawings by state-of-the-art supervised deep networks is recently shown with puzzling low performances on the SPARE3D dataset. Based on the fact that self-supervised learning is helpful when a large number of data are available, we propose two self-supervised learning approaches to improve the baseline performance for view consistency reasoning and camera pose reasoning tasks on the SPARE3D dataset. For the first task, we use a self-supervised binary classification network to contrast the line drawing differences between various views of any two similar 3D objects, enabling the trained networks to effectively learn detail-sensitive yet view-invariant line drawing representations of 3D objects. For the second type of task, we propose a self-supervised multi-class classification framework to train a model to select the correct corresponding view from which a line drawing is rendered. Our method is even helpful for the downstream tasks with unseen camera poses. Experiments show that our method could significantly increase the baseline performance in SPARE3D, while some popular self-supervised learning methods cannot.


## Data
You can download the dataset via [our google drive link](https://drive.google.com/drive/u/0/folders/1yPu3pa57eCm2iRx6AwWnp_IZK9r_RAxY). This google drive folder contains two files:
1. contrastive_spatial_reasoning.7z, which contains "contrastive data" and "supervised data". "contrastive data" is for contrastive spatial reasoning method, "supervised data" is for fine tuning.
2. In contrastive_model folder, you can find the trained model using our contrastive spatial reasoning method. Learning rate is 5e-05.

## Dependencies & Our code 
Requires Python3.x, PyTorch, PythonOCC. Running on GPU is highly recommended. The code has been tested on Python 3.8.5, PyTorch 1.8.0, with CUDA 11.1.

### Task code
We significantly improve the SPARE3D task performance. Specifically, we design a contrastive spatial reasoning method for the T2I task.
Code of three tasks(T2I, I2P, P2I) can be found under [Tasks](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks) folder.
#### [Isometric to pose](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Isometric_to_pose)
Run ```I2P_trainer.py``` with the parameters explained in args in the code. Our exploration experiments are under [structure_explore folder](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Isometric_to_pose/Structure_explore). ```network_structure``` contains the controlled experiments on network structure, such as w/o adaptive pooling layer, dropout layer, fully connected layer, and whether use ImageNet pre-trained parameters. ```network_capacity``` contains the controlled experiments on width and depth of the baseline network.
#### [Pose to isometric](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Pose_to_isometric)
Run ```P2I_trainer.py```with the parameters explained in args in the code.
#### [Three_view_to_isometric](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Three_view_to_isometric)
[contrastive_spatial_reasoning](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Three_view_to_isometric/contrastive_spatial_reasoning) folder contains the code for contrastive spatial reasoning method. Run ```Three2I_trainer.py``` with parameters under [Contrastive_learning folder](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Three_view_to_isometric/Ours/Contrastive_learning). To fine tune the network, run```Three2I_opt2_trainer.py``` with parameters under [Fine_tune folder](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Three_view_to_isometric/Ours/Fine_tune).

### Attention map generation
#### [attention_map_generation](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Data_generation/Attention_Map)
Generate attention maps of the trained model using ```attention_map.py``` with image path and the root of trained model path.
## [Paper (arXiv)](https://arxiv.org/abs/2104.13433)
To cite our paper:
```
@misc{xiang2021contrastive,
      title={Contrastive Spatial Reasoning on Multi-View Line Drawings}, 
      author={Siyuan Xiang and Anbang Yang and Yanfei Xue and Yaoqing Yang and Chen Feng},
      year={2021},
      eprint={2104.13433},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```

## Acknowledgment
The research is supported by NSF Future Manufacturing program under EEC-2036870. Siyuan Xiang gratefully thanks the IDC Foundation for its scholarship. We also thank the anonymous reviewers for constructive feedback.
