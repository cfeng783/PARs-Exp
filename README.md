# PARs-Exp
This repository includes the scripts for running experiments in the papaper: 

Cheng Feng. 2024. PARs: Predicate-based Association Rules for Efficient and
Accurate Anomaly Explanation. In Proceedings of the 33rd ACM International
Conference on Information and Knowledge Management (CIKM ’24), October
21–25, 2024, Boise, ID, USA. ACM, New York, NY, USA, 10 pages. https:
//doi.org/10.1145/3627673.3679625



#### Data preparation

Download dataset files from https://github.com/Minqi824/ADBench/tree/main/datasets/Classical, then put them under folder datasets/npz/



#### Run experiments for efficiency and accuracy study of explanation rules
```shell
cd experiments
python main_rel_eff_study.py --admodel <model>
```
Specify mode to one of the following: IF, AE



#### Run experiments for study on abnormal feature identification accuracy
```shell
cd experiments
python main_abnormal_feat_identification.py --admodel <model>
```

#### Run experiments for PoF study and ablation study
```shell
cd experiments
python main_PoF.py --admodel <model> --pm <predicate mode>
```
specify pm to one of the following: 0, 1, 2

0: Dependency-based

1: KMeans Bins

2: Uniform Bins

