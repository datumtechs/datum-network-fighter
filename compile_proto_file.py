import glob
import os
import sys

sdk_path = sys.executable
proto_root_path = './armada-common'
proto_include_path1 = os.path.join(proto_root_path, 'Fighter')
proto_include_path2 = os.path.join(proto_root_path, 'Carrier')

protos_to_compile1 = os.path.join(proto_include_path1, '*.proto')
protos_to_compile2 = os.path.join(proto_include_path2, 'lib/api/*.proto')

if not os.path.exists('./protos'):
    os.makedirs('./protos')

base_cmd = f"{sdk_path} -m grpc_tools.protoc --python_out=./protos --grpc_python_out=./protos"

cmd1 = f'{base_cmd} -I{proto_include_path1} -I{proto_root_path} {protos_to_compile1}'
print(cmd1)
os.system(cmd1)
cmd2 = f'{base_cmd} -I{proto_include_path2} -I{proto_root_path} {protos_to_compile2}'
print(cmd2)
os.system(cmd2)
