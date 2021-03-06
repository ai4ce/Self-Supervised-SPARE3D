
# Contrastive Spatial Reasoning on Multi-View Line Drawings

[**Siyuan Xiang**](https://www.linkedin.com/in/siyuanxiang/), [**Anbang Yang**](https://www.linkedin.com/in/anbang-yang/), [**Yanfei Xue**](https://www.linkedin.com/in/yanfei-xue-a6aa511b2/), [**Yaoqing Yang**](https://sites.google.com/site/yangyaoqingcmu/), [**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng)

![Overview](https://raw.githubusercontent.com/ai4ce/Contrastive-SPARE3D/main/docs/figs/teaser_fig_cut.jpg?token=AKI7ZKME7LQP3J77RHTC6R3ASRRXK)


|[Abstract](#abstract)|[Code](#code-github)|[Paper](#paper-arxiv)|[Results](#results)|[Acknowledgment](#acknowledgment)|

## Abstract
Spatial reasoning on multi-view line drawings by state-of-the-art supervised deep networks is recently shown with puzzling low performances on the SPARE3D dataset. Based on the fact that self-supervised learning is helpful when a large number of data are available, we propose two self-supervised learning approaches to improve the baseline performance for <em>view consistency reasoning</em> and <em>camera pose reasoning</em> tasks on the SPARE3D dataset. For the first task, we use a self-supervised binary classification network to contrast the line drawing differences between various views of any two similar 3D objects, enabling the trained networks to effectively learn detail-sensitive yet view-invariant line drawing representations of 3D objects. 
For the second type of task, we propose a self-supervised multi-class classification framework to train a model to select the correct corresponding view from which a line drawing is rendered. Our method is even helpful for the downstream tasks with unseen camera poses. Experiments show that our method could significantly increase the baseline performance in SPARE3D, while some popular self-supervised learning methods cannot.

## [Code (GitHub)](https://github.com/ai4ce/Contrastive-SPARE3D)
```
The code is copyrighted by the authors. Permission to copy and use this software for noncommercial use is hereby granted provided: (a) this notice is retained in all copies, (2) the publication describing the method (indicated below) is clearly cited, and (3) the distribution from which the code was obtained is clearly cited. For all other uses, please contact the authors.
The software code is provided "as is" with ABSOLUTELY NO WARRANTY expressed or implied. Use at your own risk.

This code provides an implementation of the method described in the following publication: 
Siyuan Xiang, Anbang Yang, Yanfei Xue, Yaoqing Yang, and Chen Feng    
"Contrastive Spatial Reasoning on Multi-View Line Drawings (arXiv)". 
``` 
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

### Controlled experiments for CAD model complexity
**CSG models with different number of primitives**
![env](https://raw.githubusercontent.com/ai4ce/Contrastive-SPARE3D/main/docs/figs/model_complexity_examples.JPG?token=AKI7ZKNUMDRUPT3UVKE25FDASRSEK)

**CAD model complexity vs. test accuracy**
![rel](https://raw.githubusercontent.com/ai4ce/Contrastive-SPARE3D/main/docs/figs/CSG_results.JPG?token=AKI7ZKOWZZFGDL77MUBSQGDASRSOM)

### Controlled experiments for network capacity
**Network capacity (width and depth) vs. test accuracy**
![rel](https://raw.githubusercontent.com/ai4ce/Contrastive-SPARE3D/main/docs/figs/width_depth_cropped.JPG?token=AKI7ZKNWOF2EYJ7LLOMT6TLASRSTS)


### Contrastive spatial reasoning for task T2I 
**Network architecture**
![net](https://raw.githubusercontent.com/ai4ce/Contrastive-SPARE3D/main/docs/figs/network_architecture_simple_model.JPG?token=AKI7ZKILZXQLTMR24WAYBXDASRSZW)

### Results
**Our method vs. other baseline methods**
![Baseline_curve](https://raw.githubusercontent.com/ai4ce/Contrastive-SPARE3D/main/docs/figs/contrastive_vs_other.JPG?token=AKI7ZKKVE56N2Y7MK5JQA7LASRTA4)

### Attention maps for our method vs. supervised learning
![attn](https://raw.githubusercontent.com/ai4ce/Contrastive-SPARE3D/main/docs/figs/attention_example.JPG?token=AKI7ZKL7NU7MPKBDTEYNXVDASRTF6)

## Acknowledgment
The first two authors contributed equally. Chen Feng is the corresponding author. The research is supported by NSF Future Manufacturing program under EEC-2036870. Siyuan Xiang gratefully thanks the IDC Foundation for its scholarship. 
