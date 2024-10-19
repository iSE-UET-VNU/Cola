# COLA: Leveraging Local and Global Relationships for Corrupted Label Detection     


## Introduction
The performance of the Machine Learning and Deep Learning models heavily depends on the quality and quantity of the training data. However, real-world datasets often contain a considerable percentage of noisy labels, ranging from 8.0% to 38.5%. This could significantly reduce model accuracy. 
To address the problem of corrupted labels, we propose **COLA**, a novel data-centric approach that leverages both local neighborhood similarities and global relationships across the entire dataset to detect corrupted labels. 

The main idea of our approach is that similar instances tend to share the same label, and the relationship between clean data can be learned and utilized to distinguish the correct and corrupted labels. 
Our experiments on four well-established datasets of image and text demonstrate that **COLA** consistently outperforms state-of-the-art approaches, achieving improvements of **8% to 21% in F1-score** for identifying corrupted labels across various noise types and rates. 
For visual data, **COLA** achieves improvements of up to **80% in F1-score**, while for textual data, the average improvement reaches about **17%** with a maximum of **91%**. Furthermore, **COLA** is significantly more effective and efficient in detecting corrupted labels than advanced large language models, such as **Llama3**, with improvements of up to **112% in Precision** and a **300X reduction in execution time**.

Source code for reproduce experiments can be found [here](https://github.com/gnefiew/COLA.git)

## The Architecture

![COLA-Appoarch](figs/COLA.png)

Figure illustrates the COLA approach, which consists of three main components: **Feature Representation**, **Local Verification**, and **Global Verification**. First, data instances are encoded using an embedding method to represent their features. In the Local Verification phase, the 𝑘-Nearest Neighbors (𝑘-NN) algorithm is employed to determine the labels of data based on their neighbors. Then, in the Global Verification phase, a classification model is trained on the "clean" data confirmed from the previous phase to capture general patterns and predict the remaining labels. This process can be iterated to improve accuracy.

## Environment 

## Dataset
| **Dataset** | **Data type** | **#Instances** | **#Classes** |
|-------------|---------------|----------------|--------------|
| CIFAR-10    | image         | 50.0K          | 10           |
| CIFAR-100   | image         | 50.0K          | 100          |
| Agnews      | text          | 127.6K         | 4            |
| DBPedia     | text          | 120.0K         | 14           |

## Result 
The table below summarizes the performance of **COLA**, a corrupted label detection tool, measured by F1 score across multiple datasets with varying noise types and noise rates. The datasets tested include CIFAR-10, CIFAR-100, Agnews, and DBPedia. The table evaluates COLA's effectiveness in identifying corrupted labels under the following noise type:
- **Symmetric noise (_Symm._)**: This scenario assumes incorrect labels are uniformly distributed across all possible classes. To generate _Symm._ noise, the true label of an instance is randomly flipped to one of the other classes with equal probability.

- **Asymmetric noise (_Asym._)**: In the _Asym._ scenario, incorrect labels are distributed unevenly across classes. This means some classes are more likely to be mislabeled as specific other classes.

- **Instance-dependent noise (_Inst._)**: In this scenario, the probability of an instance being mislabeled depends on its features. The idea is that instances that share similarities with other classes are prone to be mislabeled.

[Experiment Result Details](https://docs.google.com/spreadsheets/d/1OnTSKVYbpahb4R6R-A-Iu4laEvShN_OjCALj18Qwyc4/edit?usp=sharing)