import glob
import os
import sys

sdk_path = sys.executable
proto_root_path = './armada-common'
out_dir = './protos'

protos_to_compile = {'Fighter': '*.proto',
                     'Carrier': 'lib/api/*.proto',
                     'google': 'api/*.proto'}

exclude_include = ['google']

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
init_py = os.path.join(out_dir, '__init__.py')
if not os.path.exists(init_py):
    with open(init_py, 'w') as f:
        pass

base_cmd = f'{sdk_path} -m grpc_tools.protoc --experimental_allow_proto3_optional --python_out={out_dir} --grpc_python_out={out_dir}'

for proto, file in protos_to_compile.items():
    include_path = os.path.join(proto_root_path, proto)
    ff = os.path.join(include_path, file)
    cmd = f'{base_cmd} -I{include_path} -I{proto_root_path} {ff}'
    if proto in exclude_include:
        cmd = f'{base_cmd} -I{proto_root_path} {ff}'
    print(cmd)
    os.system(cmd)
