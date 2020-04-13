# FIT: Inspect Vulnerabilities in Cross-Architecture Firmware by Deep Learning and Bipartite Matching

## Environment

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
``python3 filter.py --load_path ./saved_model/405/graphnn-model_best --w2v_model ../Instruction_Embedding/myModel/ --top_similar 50 --check_dir ./suspicious/``
4. FUTURE WORK: BETTER WAY FOR FEATURE FUSION AND OF COURSE BETTER MODEL

## Graph_Match
1. Run run_graphMatch.py, find the vulnerable functions' name printed in the terminal.
``python3 run_graphMatch --sus_dir ./suspicious --json_dir ../dataset/ --threashold 1.5``
2. FUTURE WORK: BETTER BIPATITIE ALGORITHM OR DYNAMIC ANALYSIS