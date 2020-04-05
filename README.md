# FIT: Inspect Vulnerabilities in Cross-Architecture Firmware by Deep Learning and Bipartite Matching

## 3LACFG_Generator
1. In run_ida_preprocess.py, config *ida_path* and *binary_dir*, which is your ida_bin path and the binaries dir path, respectively.
2. In my_preprocess.py, config *path*, which is the output dir to store the generated *.ida files.
3. Run run_ida_preprocess.py and you will get the responding *.ida files, this will take a while...
4. In 2json.py, config *dirpath*, which might be the same as *path* in Step 2, run and get the *.json files in "./json/"

## Instruction_Embedding
