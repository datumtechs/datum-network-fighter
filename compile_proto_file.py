import sys
import os

sdk_path = sys.executable
proto_file_path = './armada-common/Fighter'
if not os.path.exists('./armada-common/Fighter/google'):
    os.system('cp -r ./armada-common/google ./armada-common/Fighter')
if not os.path.exists('./protos'):
    os.makedirs('./protos')
base_cmd = f"{sdk_path} -m grpc_tools.protoc -I{proto_file_path} --python_out=./protos  --grpc_python_out=./protos"
for file_name in os.listdir(proto_file_path):
    abs_path = os.path.join(proto_file_path, file_name)
    if os.path.isfile(abs_path):
        protoc_cmd = f'{base_cmd} {abs_path}'
        print(protoc_cmd)
        os.system(protoc_cmd)
