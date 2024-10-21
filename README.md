# COLA: Leveraging Local and Global Relationships for Corrupted Label Detection     


## Introduction
COLA is a powerful, data-centric tool that detects and eliminates corrupted labels in real-world datasets. Leveraging both local neighborhood similarities and global relationships across the entire dataset, COLA identifies noisy labels that can significantly reduce model accuracy, especially in Machine Learning and Deep Learning models.

## The Architecture

![COLA-Appoarch](figs/COLA.png)

The figure illustrates the COLA approach, which consists of three main components: **Feature Representation**, **Local Verification**, and **Global Verification**. First, data instances are encoded using an embedding method to represent their features. In the Local Verification phase, the 𝑘-Nearest Neighbors (𝑘-NN) algorithm is employed to determine the data labels based on their neighbors. Then, in the Global Verification phase, a classification model is trained on the "clean" data confirmed from the previous phase to capture general patterns and predict the remaining labels. This process can be iterated to improve accuracy.

## Results
The following link gives more detailed information about the results of the experiments

[Experiment Result Details](https://docs.google.com/spreadsheets/d/1OnTSKVYbpahb4R6R-A-Iu4laEvShN_OjCALj18Qwyc4/edit?usp=sharing)

## Quick Start
### Prerequisites
```bash
$ python3 -m venv cola
$ source cola/bin/activate
$ pip install -r requirements.txt
```
### Running the Script

#### Ag News
```bash
$ python3 main.py --data_type text --dataset_name ag_news --noise_type sym --error_rate 0.05 --encode_model bert-base-uncased
```

#### Cifar10
```bash
$ python3 main.py --data_type image --dataset_name cifar10 --noise_type sym --error_rate 0.05 --encode_model facebook/dinov2-base
```
We recommend using dinov2-base for better performance, however, for reproducing experiment results you can also download the data embedded by CLIP [here](https://drive.google.com/drive/folders/1b6U_x3NzdXa7tc23CkN5aEMdHS44kVhX?usp=sharing)

### Contact us
If you have any questions, comments, or suggestions, please do not hesitate to contact us.
- Email: 22028164@vnu.edu.vn
