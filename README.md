# EasIFA
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) <br>
Implementation of enzyme catalytic acitve site prediction with EasIFA<br><br>
![EasIFA](./img/main-model_structures.png)


## Contents

- [Publication](#publication)
- [Web Server](#web-server)
- [Quickly Start From Gitpod](#quickly-start-from-gitpod)
- [OS Requirements](#os-requirements)
- [Python Dependencies](#python-dependencies)
- [Installation Guide](#installation-guide)
- [Reproduce Results](#reproduce-results)
    - [Download Checkpoints and Dataset](#1-download-checkpoints-and-dataset)
    - [Test EasIFA](#2-test-easifa)
- [Cite Us](#cite-us)

## Publication
Multi-Modal Deep Learning Enables Ultrafast and Accurate Annotation of Enzymatic Active Sites

## Web Server

We have developed a [WebServer](http://easifa.iddd.group) for EasIFA, which allows you to conveniently annotate the active sites of enzymes you are interested in. The workflow is divided into two logical steps: [1)](http://easifa.iddd.group/from_structure) You can directly upload the PDB structure of the enzyme and the catalyzed reaction equation, [2)](http://easifa.iddd.group/from_uniprot) Provide the UniProt ID of the enzyme of interest directly.<br>

![GUI](img/Web-GUI.png)

## Quickly Start From Gitpod
About 4 minutes.<br>
[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/github.com/wangxr0526/EasIFA) 


## OS Requirements
This repository has been tested on **Linux**  operating systems.

## Python Dependencies
* Python (version >= 3.8) 
* PyTorch (version >= 1.12.1) 
* RDKit (version >= 2019)
* TorchDrug (version == 0.2.1)
* fair-esm (version == 2.0.1)
* Py3Dmol (version ==2.0.3)

## Installation Guide
Create a virtual environment to run the code of EasIFA.<br>
It is recommended to use conda to manage the virtual environment.The installation method for conda can be found [here](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html#installing-on-linux).<br>
Make sure to install pytorch with the cuda version that fits your device.<br>
This process usually takes few munites to complete.<br>
```
git clone https://github.com/wangxr0526/EasIFA.git
cd EasIFA
conda env create -f envs.yml
conda activate easifa_env
```


## Reproduce Results
### **[1]** Download Checkpoints and Dataset

Running the following command can download the model's checkpoints and datasets (including the PDB structures in the dataset).

```
python download_data.py
```
The links correspond to the paths of the zip files as follows:
```
https://drive.google.com/uc?id=1pf0pMNELXYR9yU_w3ZkJL0rjbjqHBQaB    --->    checkpoints.zip  (7.1Gb)
https://drive.google.com/uc?id=15c-KoZ47TpF9_qyQfJiY67gcgVZ8N5WR    --->    dataset.zip      
```

### **[2]** Test EasIFA
Test in the SwissProt E-RXN ASA dataset:

Active site position prediction task:

```
python main_test.py --gpu CUDA_ID \
                    --task_type active-site-position-prediction \
                    --dataset_path dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100 \
                    --checkpoint checkpoints/enzyme_site_predition_model/train_in_uniprot_ecreact_cluster_split_merge_dataset_limit_100_at_2023-06-07-16-24-32/global_step_27000
```
Active site categorie prediction task

```
python main_test.py --gpu CUDA_ID \
                    --task_type active-site-categorie-prediction \
                    --dataset_path dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100 \
                    --checkpoint checkpoints/enzyme_site_type_predition_model/train_in_uniprot_ecreact_cluster_split_merge_dataset_limit_100_at_2023-06-14-11-04-55/global_step_70000
```
Test in the MCSA E-RXN CSA dataset:
```
python test_knowledge_transfer_learning.py --gpu CUDA_ID \
                                            --dataset_path dataset/mcsa_fine_tune/normal_mcsa \
                                            --structure_path dataset/mcsa_fine_tune/structures \
                                            --checkpoint checkpoints/enzyme_site_type_predition_model/checkpoints/enzyme_site_predition_model_finetune_with_mcsa/train_in_normal_mcsa_at_2023-10-06-09-48-04/global_step_37200
```
## Cite Us

Under review
