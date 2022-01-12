import os
import re
import sys
import shutil

sdk_path = sys.executable
proto_root_path = './armada-common'
out_dir = './protos'
v3 = True
protos_to_compile = {'Fighter': 'lib/*.proto',
                     'google': 'api/*.proto',
                     'Carrier': 'lib/common/*.proto'}

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

pat = re.compile(r'^import\s+"(.+)";')


def find_deps(proto, base_dir, depth=1):
    file = os.path.join(base_dir, proto)
    with open(file, encoding='utf8') as f:
        lines = f.readlines()
    deps = set()
    for i in lines:
        m = re.match(pat, i)
        if m:
            proto1 = m.group(1)
            if proto1.startswith('google') or proto1.startswith('lib/common'):
                continue
            deps.add(proto1)
    for proto1 in deps.copy():
        its_deps = find_deps(proto1, base_dir, depth + 1)
        deps.update(its_deps)
    return deps


include_path1 = os.path.join(proto_root_path, 'Carrier')
include_path2 = os.path.join(include_path1, 'lib/types')
include_path3 = os.path.join(include_path1, 'lib/api')

ref_carrier_proto = 'lib/api/sys_rpc_api.proto'
deps = find_deps(ref_carrier_proto, include_path1)
deps.add(ref_carrier_proto)

for proto in deps:
    ff = os.path.join(include_path1, proto)
    cmd = f'{base_cmd} -I{include_path1} -I{proto_root_path} {ff}'
    print(cmd)
    os.system(cmd)

if v3:
    shutil.rmtree('./lib')
    shutil.copytree(f'{out_dir}/lib', './lib')
    shutil.copytree(f'{out_dir}/google', './lib/google')
    shutil.copyfile(f'{out_dir}/__init__.py', './lib/__init__.py')
    shutil.copyfile(f'{out_dir}/__init__.py', './lib/api/__init__.py')
    shutil.copyfile(f'{out_dir}/__init__.py', './lib/common/__init__.py')
    shutil.copyfile(f'{out_dir}/__init__.py', './lib/google/__init__.py')
    shutil.copyfile(f'{out_dir}/__init__.py', './lib/types/__init__.py')
    shutil.rmtree(out_dir)
