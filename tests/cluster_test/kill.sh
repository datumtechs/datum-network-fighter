scripts_path=$(cd $(dirname $0); pwd)
log_path=${scripts_path}"/log"
nodes_conf=${scripts_path}"/nodes_conf"
base_dir=${scripts_path}"/../.."
python_command=python3

cd $base_dir
# kill
${python_command} tools/onekey_deploy.py --kill_all $nodes_conf/nodes_conf.json

# clean directory
