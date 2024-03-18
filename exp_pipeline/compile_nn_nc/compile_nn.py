from neural_cast.compiler import run
from neural_cast.frontend.common.common import CompilerConfig
import os
import yaml
import sys

onnx_name : str = sys.argv[1]

curr_file = os.path.abspath(__file__)
curr_path = os.path.dirname(curr_file)
with open(curr_path + '/config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
CompilerConfig(config)

# init config file
workdir : str = CompilerConfig()['workdir']
path_onnx : str = workdir + 'onnx_files/' + onnx_name

# run compiler
run(CompilerConfig(), framework='onnx', path=path_onnx)