import glob
import os
import sys

sdk_path = sys.executable
proto_file_path = './armada-common/Fighter'
if not os.path.exists('./armada-common/Fighter/google'):
    os.system('cp -r ./armada-common/google ./armada-common/Fighter')
if not os.path.exists('./protos'):
    os.makedirs('./protos')

base_cmd = f"{sdk_path} -m grpc_tools.protoc -I{proto_file_path} --python_out=./protos  --grpc_python_out=./protos"
for file_name in glob.glob(proto_file_path + '/**/*.proto', recursive=True):
    protoc_cmd = f'{base_cmd} {file_name}'
    print(protoc_cmd)
    os.system(protoc_cmd)
