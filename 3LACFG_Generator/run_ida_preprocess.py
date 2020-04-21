import os
import subprocess

# ida app path
# ida_path = '/Applications/IDA\ Pro\ 7.0/ida.app/Contents/MacOS/ida64'
ida_path = '/Applications/ida.app/Contents/MacOS/ida64'
# the dir of binary file to be analyzed
# binary_dir = '/Users/eacials/Downloads/openssl'
binary_dir = '../Dataset/CoreUtils/coreutils_6.12_arm_O0_bin/'
# the ida plugin script path
# script_path = './my_preprocess.py'
script_path = '~/Desktop/FIT/3LACFG_Generator/my_preprocess.py'
for file in os.listdir(binary_dir):
    # cmd_str = ida.exe -Lida.log -c -A -Sanalysis.py pefile
    cmd_str = '{} -Lida.log -c -A -S{} {}'.format(ida_path, script_path, os.path.join(binary_dir, file))
    print(cmd_str)
    if file.startswith('openssl'):  # binary file filter
        p = subprocess.Popen(cmd_str, shell=True)
        p.wait()