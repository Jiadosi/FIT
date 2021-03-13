# FIT: Inspect Vulnerabilities in Cross-Architecture Firmware by Deep Learning and Bipartite Matching

## Prerequiste
Make sure you have installed all of following packages or libraries (including dependencies if necessary) in your workspace:
1. Tensorflow
2. [gensim](https://radimrehurek.com/gensim/)
3. [scikit-learn](https://scikit-learn.org/stable/index.html)
4. pickle in python
5. [IDA Pro](https://www.hex-rays.com/products/ida/)

## Dataset
* OpenSSL in Instruction_Embedding/dataset/filtered_json_inst/
* CoreUtils in Dataset/CoreUtils/json/
* FindUtils in Dataset/findutils.zip
* BusyBox in Dataset/busybox.zip

## 3LACFG_Generator
1. In run_ida_preprocess.py, config *ida_path* and *binary_dir*, which is your ida_bin path and the binaries dir path, respectively.
2. In my_preprocess.py, config *path*, which is the output dir to store the generated *.ida files.
3. Run run_ida_preprocess.py and you will get the responding *.ida files, this will take a while...
4. In 2json.py, config *dirpath*, which might be the same as *path* in Step 2, run and get the *.json files in "./json/".
5. FUTURE WORK: CONTEXT SENSITIVE

## Instruction_Embedding
1. In instEmbedding.py, config *dirPath* and *filePath* which is your dataset path and the output path, run preparing(dirPath, filePath) to fetch all instructions from a particular architecture.
2. In instEmbedding.py, config *modelPath* which is the output model path, run inputGen(filePath) first and then training(modelPath, output of inputGen) to train the w2v model.
3. My trained w2v models can be found in "./myModel/".
4. FUTURE WORK: ARCHITECTURE FREE

## Block_Embedding & Graph_Embedding
1. Run __train.py, this will take a long time Orzzzzz  
``python3 __train.py --save_path ./saved_model/405/ --w2v_model ../Instruction_Embedding/myModel/``
2. The trained model will be stored in "./saved_model/". The AUC of FIT model is 0.97.
3. In filter.py, config *load_path* which is the trained model path, and *top_similar* which means top N most similar functions. Run filter.py, get the similar score between function pairs and N suspicious vulnerable function names can be found in *check__dir*. Note that, the vulnerable binary function should be the last json item in the json file which store all the preprocessed functions from the under-test binary!  
``python3 filter.py --load_path ./saved_model/405/graphnn-model_best --w2v_path ../Instruction_Embedding/myModel/ --top_similar 50 --check_dir ./suspicious/``
4. FUTURE WORK: BETTER WAY FOR FEATURE FUSION AND OF COURSE BETTER MODEL

## Graph_Match
1. Run run_graphMatch.py, find the vulnerable functions' name printed in the terminal.  
``python3 run_graphMatch.py --sus_dir ../Block_Graph_Embedding/suspicious/ --json_dir ../Instruction_Embedding/dataset/filtered_json_inst/ --threashold 1.5``
2. FUTURE WORK: BETTER BIPATITIE ALGORITHM OR DYNAMIC ANALYSIS


### Cite
If you use FIT in scientific work, consider citing [our paper](https://www.sciencedirect.com/science/article/pii/S0167404820303059) presented at COSE'20:

Bibtex:
```
@article{LIANG2020102032,
title = {FIT: Inspect vulnerabilities in cross-architecture firmware by deep learning and bipartite matching},
journal = {Computers & Security},
volume = {99},
pages = {102032},
year = {2020},
issn = {0167-4048},
doi = {https://doi.org/10.1016/j.cose.2020.102032},
url = {https://www.sciencedirect.com/science/article/pii/S0167404820303059},
author = {Hongliang Liang and Zhuosi Xie and Yixiu Chen and Hua Ning and Jianli Wang},
keywords = {firmware security, binary code, similarity detection, neural network, bipartite matching},
abstract = {Widely deployed IoT devices expose serious security threats because the firmware in them contains vulnerabilities, which are difficult to detect due to two main factors: 1) The firmware’s code is usually not available; 2) A same vulnerability often exists in multiple firmware with different architectures and/or release versions. In this paper, we propose a novel neural network-based staged approach to inspect vulnerabilities in firmware, which first learns semantics in binary code and utilizes neural network model to screen out the potential vulnerable functions, then performs bipartite graph matching upon three-level features between two binary functions. We implement the approach in a tool called FIT and evaluation results show that FIT outperforms state-of-the-art approaches, i.e., Gemini, CVSSA and discovRE, on both effectiveness and efficiency. FIT also detects vulnerabilities in real-world firmware of IoT devices, such as D-Link routers. Moreover, we make our tool and dataset publicly available in the hope of facilitating further researches in the firmware security field.}
}
```
