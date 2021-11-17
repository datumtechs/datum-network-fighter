scripts_path=$(cd $(dirname $0); pwd)

cd ${scripts_path}/nodes_conf
export PYTHONPATH=$PYTHONPATH:../../..:../../../protos:../../../common
echo "***** 1. generate deploy config"
python3 generate_nodes_conf.py

echo "***** 2. generate ssl certs"
python3 generate_ssl_ini.py
bash ../../../third_party/gmssl/gen_certs_gmssl.sh ${scripts_path}/nodes_conf/new_ssl.ini

echo "***** 3. generate task config"
python3 generate_task_cfg.py --algo_type=logistic_regression
