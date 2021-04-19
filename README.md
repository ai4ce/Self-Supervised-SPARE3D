# Contrastive Spatial Reasoning on Multi-View Line Drawings

[**Siyuan Xiang**](), [**Anbang Yang**](https://github.com/endeleze), [**Yanfei Xue**](),[**Yaoqing Yang**](),[**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng)

## Abstract
Spatial reasoning on multi-view line drawings by state-of-the-art supervised deep networks is recently shown with puzzling low performances on the SPARE3D dataset. 
To study the reason behind the low performance and to further our understandings of these geometric visual reasoning tasks, we design controlled experiments on both input data and baseline networks.
Guided by the hindsight from these experiment results, we propose a simple contrastive learning approach along with other network modifications to improve the baseline performance. 
Our approach uses a self-supervised binary classification network to compare the line drawing differences between various views of any two similar 3D objects.
It enables deep networks to effectively learn detail-sensitive yet view-invariant line drawing representations of 3D objects. 
Experiments show that our method could significantly increase the baseline performance in SPARE3D, while some other popular self-supervised learning methods cannot.
## [Code (GitHub)](https://github.com/ai4ce/SNAC) & Dependencies
All environment scripts can be found in [Env](https://github.com/ai4ce/SNAC/tree/main/Env) folder. These environments are developed based on the [OpenAi Gym](https://gym.openai.com/). All baseline scripts are in [script](https://github.com/ai4ce/SNAC/tree/main/script) floder. You need to install the [Pytorch](https://pytorch.org/) to run all baseline scripts. We use [Stable baseline](https://github.com/openai/baselines/) for PPO algorithm. 

## [Paper (arXiv)](https://arxiv.org/abs/2103.16732)
To cite our paper:
```

```

## Acknowledgment
 
