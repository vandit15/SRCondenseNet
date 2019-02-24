# SRCondenseNet

This repository contains the code (in PyTorch) for "[Efficient Single Image Super Resolution using Enhanced Learned Group Convolutions](https://arxiv.org/abs/1808.08509)", paper by Vandit Jain, Prakhar Bansal, Abhinav Kumar Singh, Rajeev Srivastava.

### Citation

If you find our project useful in your research, please consider citing:

```
@article{DBLP:journals/corr/abs-1808-08509,
  author    = {Vandit Jain and
               Prakhar Bansal and
               Abhinav Kumar Singh and
               Rajeev Srivastava},
  title     = {Efficient Single Image Super Resolution using Enhanced Learned Group
               Convolutions},
  journal   = {CoRR},
  volume    = {abs/1808.08509},
  year      = {2018},
  url       = {http://arxiv.org/abs/1808.08509},
  archivePrefix = {arXiv},
  eprint    = {1808.08509},
  timestamp = {Sun, 02 Sep 2018 15:01:55 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1808-08509},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)

## Introduction
we propose a novel SISR method that uses relatively less number of computations. On training, we get group convolutions that have unused connections removed. We have refined this system specifically for the task at hand by removing unnecessary modules from original CondenseNet. Further, a reconstruction network consisting of deconvolutional layers has been used in order to upscale to high resolution. All these steps significantly reduce the number of computations required at testing time. We evaluate the method using various benchmark datasets and show that it performs favourably against the state-of-the-art methods in terms of both accuracy and number of computations required.

## Usage

### Dependencies

- [Python3](https://www.python.org/downloads/)
- [PyTorch(0.1.12+)](http://pytorch.org)

### Training
```
python main.py --model condensenet -b 256 -j 20 /PATH/TO/DATA \
--stages 7-7-7-7 --growth 14-14-14-14
```


## Results
We have used 91 images from Yang et al.[24] and 200 images from the Berkeley Segmentation Dataset(BSD)[25] for training.

### Results on various datasets:
| Dataset | PSNR | SSIM |
|---|---|---|---|
|Set5 | 37.79 | 0.9594 |
|Set14 | 33.23 | 0.9137 |
|Urban100 | 31.24 | 0.9190 |

## Contact
jainvandit15@gmail.com
prakhar.bansal.cse15@itbhu.ac.in

