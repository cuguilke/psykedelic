# psykedelic
psykedelic: <ins>P</ins>runing <ins>Sy</ins>stem in <ins>Ke</ins>ras for a <ins>De</ins>eper <ins>L</ins>ook <ins>I</ins>nto <ins>C</ins>onvolutions

## Table of Contents

1. [Introduction](#introduction)
2. [Citation](#citation)
3. [Dependencies](#dependencies)
4. [Training](#training)
5. [Analysis](#analysis)


## Introduction

This repository contains the official codes that are used to conduct the experiments described in the paper:

> [**A Deeper Look into Convolutions via Pruning**](https://arxiv.org/abs/2102.02804)            
> [Ilke Cugu](https://cuguilke.github.io/), [Emre Akbas](https://user.ceng.metu.edu.tr/~emre/)         

## Citation

If you use these codes in your research, please cite:

```bibtex
@article{cugu2021deeper,
  title={A Deeper Look into Convolutions via Pruning},
  author={Cugu, Ilke and Akbas, Emre},
  journal={arXiv preprint arXiv:2102.02804},
  year={2021}
}
```
  
## Dependencies
- Prerequisites:
```
Python 3.7
Keras 2.2.4
Tensorflow 1.15
matplotlib 3.1.1
matplotlib_venn 0.11.5
NumPy 1.17.2
SciPy 1.3.1
scikit-learn 0.21.3
overrides 2.8.0
```

- We also include a YAML script `./dev.yml` that is prepared for an easy Anaconda environment setup. 

## Training

Training is done via run.py. To get the up-to-date list of commands:
```
python run.py --help
```

We include a sample script `./run_experiments.sh` for a quick start.

## Analysis

In order to understand the statistical significance of the empirical results, `./analysis/ExperimentRecorder.py` stores cumulative results in [experiments.json](https://www.dropbox.com/s/62utgg7z9p2fqc7/experiments.json?dl=0).

Then, we use `./analysis/ExperimentProcessor.py` to examine the accumulated empirical data:

- In order to produce complex vs. real eigenvalue distribution statistics LaTeX tables:
```
python ExperimentProcessor.py --eig_stats
```

- In order to produce compression performance LaTeX tables:
```
python ExperimentProcessor.py --performance
```

- In order to produce the set analysis charts that we include in the Appendix of the paper:
```
python ExperimentProcessor.py --set_analysis
```

- In order to produce the pruning per significance threshold charts:
```
python ExperimentProcessor.py --pruning_per_threshold
```
![...](https://github.com/cuguilke/psykedelic/blob/main/results/ThinMicroResNet_pruning_per_threshold.png?raw=true)

- In order to produce the pruning per layer charts:
```
python ExperimentProcessor.py --pruning_per_layer
``` 
![...](https://github.com/cuguilke/psykedelic/blob/main/results/ThinMicroResNet_pruning_per_layer.png?raw=true)

- In order to produce the pruning through epochs charts:
```
python ExperimentProcessor.py --performance_history
``` 
![...](https://github.com/cuguilke/psykedelic/blob/main/results/ThinMicroResNet_score_history_full.png?raw=true)
