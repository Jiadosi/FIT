import os
import subprocess

ida_path = '/Applications/IDA\ Pro\ 7.0/ida.app/Contents/MacOS/ida64'
work_dir = os.path.abspath('.')
binary_dir = '/Users/eacials/Downloads/openssl'
script_path = '/Users/eacials/Downloads/Gencoding-master/FIT/3LACFG_Generator/my_preprocess.py'
for file in os.listdir(binary_dir):
    # cmd_str = ida.exe -Lida.log -c -A -Sanalysis.py pefile
    cmd_str = '{} -Lida.log -c -A -S{} {}'.format(ida_path, script_path, os.path.join(binary_dir, file))
    print(cmd_str)
    if file.startswith('openssl'):
        p = subprocess.Popen(cmd_str, shell=True)
        p.wait()