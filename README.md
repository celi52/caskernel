# Kernel-Based Structural-Temporal Cascade Learning for Popularity Prediction
This repo provides a reference implementation of **CasKernel**

[comment]: <> ([comment]: <> &#40;>  Quantifying the Scientific Impact via Heterogeneous Dynamical Graph Neural Network  &#41;)

[comment]: <> ([comment]: <> &#40;>  [Xovee Xu]&#40;https://xovee.cn&#41;, Fan Zhou, Ce Li, Goce Trajcevski, Ting Zhong, and Kunpeng Zhang &#41;)

[comment]: <> ([comment]: <> &#40;>  Submitted for review  &#41;)

## Dataset

For Weibo, we use the data provided by [Qi Cao, Huawei Shen et al.](https://github.com/CaoQi92/DeepHawkes)

And for APS, You can access the original APS dataset [here](https://journals.aps.org/datasets). (Released by *American Physical Society*, obtained at Jan 17, 2019)

## Dependencies

Recent versions of the following packages for Python 3 are required:
```
Python==3.7
Tensorflow-gpu==2.4.0
Cuda==11.0.221
```

## Usage
###Step1: Preprocess for generating cascade
```
python preprocess.py
```
###Step2: Generate cascade
```
python generate_cas.py
```
###Step3: Training and Testing
```
python caskernel.py
```


