# Incremenal Dimensinality Reduction
This repository provides the implementation of the method proposed in our paper "Covariance-free Partial Least Squares: An Incremental Dimensionality Reduction Method"

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Incremental Dimensinality Reduction methods
Traditional dimensionality reduction methods are not suitable for large datasets (e.g., ImageNet) since it 
requires all the data to be in memory in advance, which is often impractical due to hardware limitations. Additionally, this requirement
prevents us from employing traditional dimensionality reduction methods on streaming applications, where the data are being generated continuously.
To handle this problem, many works have proposed incremental versions of traditional dimensionality reduction methods, where the idea is to 
estimate the projection matrix using a single data sample at a time while keeping some properties of the traditional dimensionality 
reduction methods.

## Requirements
- [Scikit-learn](http://scikit-learn.org/stable/)
- [Python 3](https://www.python.org/)

## Quick Start
[binary_classification.py](binary_classification.py) and [multiclass_classification.py](multiclass_classification.py) provide examples of our method 
(named Covariance-free Partial Least Squares - CIPLS) to binary and multiclass problems, respectively.

## Parameters
Our method takes a single parameter:
1. Number of components (see n_components in [binary_classification.py](binary_classification.py) and [multiclass_classification.py](multiclass_classification.py))
Please check our [paper](https://arxiv.org/abs/1910.02319) for more details regarding this parameter, as well as the parameters from other methods.

## Additional Parameters
1. Memory restricted indicates whether the data (storage in .h5 files) do not fit into memory. In this case, set the variable memory_restricted to True.
2. Batch size (required only if memory_restricted is True) indicates the number of samples loaded into memory and sent to the incremental methods. 

### Results
Tables below compare our method with existing incremental partial least squares methods in terms of accuracy and time complexity for
estimating the projection matrix. Please check our [paper](https://arxiv.org/abs/1910.02319) for more detailed results.

Comparison of existing incremental methods in terms of accuracy. The symbol '-' denotes that it was not possible to execute the method on the respective 
dataset due to memory constraints or convergence problems. PLS denotes the use of the traditional Partial Least Squares method.

| Method 	| LFW 	| YTF 	| ImageNet 	|
|:------------:	|:-----:	|:-----:	|:--------:	|
| CCIPCA 	| 89.87 	| 81.48 	| 52.58 	|
| SGDPLS 	| 90.60 	| 83.22 	| - 	|
| IPLS 	| 90.30 	| 82.22 	| 65.74 	|
| CIPLS (Ours) 	| 91.68 	| 84.10 	| 67.09 	|
| PLS 	| 92.47 	| 85.96 	| - 	|

Comparison of incremental dimensionality reduction methods in terms of time complexity for estimating the projection matrix. 
m, n denote dimensionality of the original data and number of samples, while c, L and T denote the number of PLS components, number of PCA components and convergence steps, respectively.

|  	| Time Complexity 	|
|:------------:	|:---------------:	|
| CCIPCA 	| O(nLm) 	|
| SGDPLS 	| O(Tcm) 	|
| IPLS 	| O(nLm+c²m) 	|
| CIPLS (Ours) 	| O(ncm) 	|

Please cite our [paper](https://arxiv.org/abs/1910.02319) in your publications if it helps your research.
```bash
@article{Jordao:2018,
author    = {Artur Jordao,
Maiko Lie,
Victor Hugo Cunha de Melo and
William Robson Schwartz},
title     = {Covariance-free Partial Least Squares: An Incremental Dimensionality Reduction Method},
journal = {ArXiv e-prints},
eprint={1910.02319},
}
```
We would like to thank Xue-Qiang, Zeng Guo-Zheng Li, Raman Arora, Poorya Mianjy, Alexander Stott and Teodor Marinov for sharing their source code.
