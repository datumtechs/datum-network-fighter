import os
import re
import sys
import shutil

sdk_path = sys.executable
proto_root_path = './armada-common'
out_dir = './pb'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


base_cmd = f'{sdk_path} -m grpc_tools.protoc --experimental_allow_proto3_optional --python_out={out_dir} --grpc_python_out={out_dir}'
# the proto except carrier
protos_to_compile = [
    'fighter/*/*.proto',
    'fighter/*/*/*.proto',
    'google/api/*.proto',
    'common/*/*.proto'
]

# the carrier proto with its dependence needed by fighter
pattern = re.compile(r'^import\s+"(.+)";')
def find_deps(proto, base_dir, depth=1):
    file = os.path.join(base_dir, proto)
    with open(file, encoding='utf8') as f:
        lines = f.readlines()
    deps = set()
    for i in lines:
        m = re.match(pattern, i)
        if m:
            proto1 = m.group(1)
            if proto1.startswith('google') or proto1.startswith('common'):
                continue
            deps.add(proto1)
    for proto1 in deps.copy():
        its_deps = find_deps(proto1, base_dir, depth + 1)
        deps.update(its_deps)
    return deps
ref_carrier_proto = 'carrier/api/sys_rpc_api.proto'
deps = find_deps(ref_carrier_proto, proto_root_path)
deps.add(ref_carrier_proto)

protos_to_compile.extend(list(deps))
# compile all proto
for proto in protos_to_compile:
    proto_file = os.path.join(proto_root_path, proto)
    cmd = f'{base_cmd} -I{proto_root_path} {proto_file}'
    print(cmd)
    os.system(cmd)
print("compile all proto success.")


def mk_init_file(base_path):
    init_py = os.path.join(base_path, '__init__.py')
    if not os.path.exists(init_py):
        with open(init_py, 'w') as f:
            pass
# add init file to every folder
for dir_ in os.walk(out_dir):
    mk_init_file(dir_[0])
print("add init file to every folder success.")
