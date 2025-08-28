# SSF-Strategic-Selection-and-Forgetting
This is the code for the paper: ["Continual Learning with Strategic Selection and Forgetting for Network Intrusion Detection"]([https://ieeexplore.ieee.org/document/10621346/](https://arxiv.org/abs/2412.16264)) (Infocom 2025) 

Xinchen Zhang, Running Zhao, Zhihan Jiang, Handi Chen, Yulong Ding, Edith C.H. Ngai, Shuang-hua Yang.

## Dependencies
The project is implemented using PyTorch and has been tested on the following hardware and software configuration:

- Ubuntu 20.04 Desktop
- NVIDIA GeForce RTX 3090 GPUs
- CUDA, version = 11.7
- PyTorch, version = 1.13.1
- Anaconda3

### Installation
To install the necessary libraries and dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Experiments
We tested the effectiveness of our proposed method on the NSL-KDD and UNSW-NB15 datasets. Preprocessed versions of these datasets are provided in this repository, allowing for immediate execution. The continuous attributes have been normalized, and categorical attributes have been one-hot encoded.

Here is two examples for each dataset (NSL-KDD and UNSW-NB15) of how to start training:
```bash
python ssf.py --dataset nsl --epochs 200 --epoch_1 20 --sample_interval 5000 --num_labeled_sample 50 --opt_old_lr 100 --opt_new_lr 8 --new_sample_weight 3
```
```bash
python ssf.py --dataset unsw --epochs 200 --epoch_1 180 --sample_interval 20000 --num_labeled_sample 200 --opt_old_lr 24 --opt_new_lr 50 --new_sample_weight 60 
```

## Citation
If you find this code useful in your research, please cite:
```bibtex
@inproceedings{zhang2025continual,
  title={Continual learning with strategic selection and forgetting for network intrusion detection},
  author={Zhang, Xinchen and Zhao, Running and Jiang, Zhihan and Chen, Handi and Ding, Yulong and Ngai, Edith CH and Yang, Shuang-Hua},
  booktitle={IEEE INFOCOM 2025-IEEE Conference on Computer Communications},
  pages={1--10},
  year={2025},
  organization={IEEE}
}
```


