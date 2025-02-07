# SSF-Strategic-Selection-and-Forgetting
This is the code for the paper: ["Continual Learning with Strategic Selection and Forgetting for Network Intrusion Detection"](link-to-my-paper) (Infocom 2025)  
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

Here is an example of how to start training:
```bash
python ssf.py --dataset unsw --epochs=200 --epoch_1=180 --sample_interval 20000 --num_labeled_sample 200 --opt_old_lr 24 --opt_new_lr 50 --new_sample_weight 65 
```

## Citation
If you find this code useful in your research, please cite:
```bibtex
@article{zhang2024continual,
  title={Continual Learning with Strategic Selection and Forgetting for Network Intrusion Detection},
  author={Zhang, Xinchen and Zhao, Running and Jiang, Zhihan and Chen, Handi and Ding, Yulong and Ngai, Edith CH and Yang, Shuang-Hua},
  journal={arXiv preprint arXiv:2412.16264},
  year={2024}
}
```


