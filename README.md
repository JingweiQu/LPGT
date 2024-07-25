# Graph Transformer for Label Placement

Air Conditioner | Dishwasher | Remote    
:--:|:--:|:--:
<img src="examples/air conditioner.jpg" alt="Ground-truth vs. LPGT" height="150" /> | <img src="examples/dishwasher.jpg" alt="Ground-truth vs. LPGT" height="150" /> | <img src="examples/remote.jpg" alt="Ground-truth vs. LPGT" height="150" />

This repository is the implementation of the paper: 

Jingwei Qu, Pingshun Zhang, Enyu Che, Yinan Chen, and Haibin Ling. [Graph Transformer for Label Placement](https://jingweiqu.github.io/project/LPGT/index.html). *(TVCG)*

It contains the training and evaluation procedures in the paper.

## Requirements
* **[Python](https://www.python.org/)** (>= 3.10.12)
* **[PyTorch](https://pytorch.org/)** (>= 2.0.1)
* **[PyG](https://www.pyg.org/)** (>= 2.3.1)

## Dataset
Download the [SWU-AMIL](https://higa.teracloud.jp/share/11e16e39781d2703) dataset and extract it to the folder `data`.

## Evaluation
Download the [trained model](https://higa.teracloud.jp/share/11e103bcb9e85fe7) into the folder `trained_models`. Then Run evaluation:
```bash
python test.py experiments/amil.json
```

## Training
Run training:
```bash
python train.py experiments/amil.json
```

## Citation
```text
@article{qu2024graph,
 title={Graph Transformer for Label Placement},
 author={Qu, Jingwei and Zhang, Pingshun and Che, Enyu and Chen, Yinan and Ling, Haibin},
 journal={IEEE Transactions on Visualization and Computer Graphics},
 year={2024}
}
```
