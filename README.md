# Contrastive Spatial Reasoning on Multi-View Line Drawings

[**Siyuan Xiang**](), [**Anbang Yang**](https://github.com/endeleze), [**Yanfei Xue**](),[**Yaoqing Yang**](),[**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng)

## Abstract
Spatial reasoning on multi-view line drawings by state-of-the-art supervised deep networks is recently shown with puzzling low performances on the SPARE3D dataset. To study the reason behind the low performance and to further our understandings of these geometric visual reasoning tasks, we design controlled experiments on both input data and baseline networks.
Guided by the hindsight from these experiment results, we propose a simple contrastive learning approach along with other network modifications to improve the baseline performance. 
Our approach uses a self-supervised binary classification network to compare the line drawing differences between various views of any two similar 3D objects.
It enables deep networks to effectively learn detail-sensitive yet view-invariant line drawing representations of 3D objects. 
Experiments show that our method could significantly increase the baseline performance in SPARE3D, while some other popular self-supervised learning methods cannot.
## [Code (GitHub)](https://github.com/ai4ce/SNAC) & Dependencies
### Tasks code
Code of three Tasks can be found in [Tasks](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks), for each task:
#### [Isometric to pose](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Isometric_to_pose)
You can directly run ```I2P_trainer.py``` in command line with parameters to run our pipline. Our exploration experiments are under [structure_explore folder](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Isometric_to_pose/Structure_explore). ```Different_factors``` contains the experiments on different factors such like adaptive pooling layer, dropout layer, fully connected layer and so on that may have influence on network performance. ```Different_width_depth``` contains the experiments on width and depth of network that may have influence on performance.
#### [Pose to isometric](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Pose_to_isometric)
You can directly run ```P2I_trainer.py``` in command line with parameters to run our pipline.
#### [Three_view_to_isometric](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Three_view_to_isometric)
Under [Ours](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Three_view_to_isometric/Ours) folder contains our pipline, you can do contrastive learning experiment to run ```Three2I_trainer.py``` with parameters under [Contrastive_learning folder](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Three_view_to_isometric/Ours/Contrastive_learning) and fine tune the network using ```Three2I_opt2_trainer.py``` with parameters under [Fine_tune folder](https://github.com/ai4ce/Contrastive-SPARE3D/tree/main/Tasks/Three_view_to_isometric/Ours/Fine_tune).
### Data generation

## [Paper (arXiv)](https://arxiv.org/abs/2103.16732)
To cite our paper:
```

```

## Acknowledgment
The research is supported by NSF Future Manufacturing program under EEC-2036870. Siyuan Xiang gratefully thanks the IDC Foundation for its scholarship. We also thank the anonymous reviewers for constructive feedback.
