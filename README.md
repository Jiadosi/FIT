# FIT: Inspect Vulnerabilities in Cross-Architecture Firmware by Deep Learning and Bipartite Matching

## 3LACFG_Generator
1. In run_ida_preprocess.py, config *ida_path* and *binary_dir*, which is your ida_bin path and the binaries dir path, respectively.
2. In my_preprocess.py, config *path*, which is the output dir to store the generated *.ida files.
3. Run run_ida_preprocess.py and you will get the responding *.ida files, this will take a while...
4. In 2json.py, config *dirpath*, which might be the same as *path* in Step 2, run and get the *.json files in "./json/"
5. FUTURE WORK: CONTEXT SENSITIVE

## Instruction_Embedding
1. In instEmbedding.py, config *dirPath* and *filePath* which is your dataset path and the output path, run preparing(dirPath, filePath) to fetch all instructions from a particular architecture.
2. In instEmbedding.py, config *modelPath* which is the output model path, run inputGen(filePath) first and then training(modelPath, output of inputGen) to train the w2v model.
3. My trained w2v models can be found in "./myModel/".
3. FUTUREWORK: ARCHITECTURE FREE

## Block_Embedding & Graph_Embedding

## Graph_Match