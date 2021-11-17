scripts_path=$(cd $(dirname $0); pwd)
log_path=${scripts_path}/log
nodes_conf=${scripts_path}/nodes_conf
base_dir=${scripts_path}/../..

environment_name=new_py37
remote_dir='~/fighter'
py_home=${remote_dir}/${environment_name}

cd $base_dir
echo "***** 1. deploy source code & cfg"
src_zip=fighter.tar.gz
tar -czPf $src_zip common protos data_svc compute_svc third_party/via_svc tests console gateway third_party/gmssl algorithms
python3 tools/onekey_deploy.py --src_zip=$src_zip --py_home=$py_home $nodes_conf/nodes_conf.json

echo "***** 2. deploy environment & certs. please waiting..."
py_env_zip=${environment_name}.tar.gz
cp -r ~/miniconda3/envs/${environment_name} .
tar -czPf $py_env_zip ${environment_name}
certs_zip=certs.tar.gz
tar -czPf $certs_zip certs
python3 tools/onekey_deploy.py --py_env_zip=$py_env_zip --certs_zip=$certs_zip $nodes_conf/nodes_conf.json

echo "***** 3. start data&compute&schedule service"
python3 tools/onekey_deploy.py --start_all $nodes_conf/nodes_conf.json
