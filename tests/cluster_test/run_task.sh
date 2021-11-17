
export PYTHONPATH=$PYTHONPATH:..:../protos:../common
mkdir -p log
scripts_path=$(cd $(dirname $0); pwd)
base_dir=${scripts_path}"/../.."
cfg=config.yaml
ip=$(cat nodes_conf/config.yaml | shyaml get-value ip.0)
data_svc_base_port=$(cat nodes_conf/config.yaml | shyaml get-value data_svc_port)

cd $base_dir/console
echo "start console that connect to data_svc which internal port ${data_svc_base_port}"
echo "run task command:  comp_run_task <task_id> <task_cfg_file>"
python3 main.py --config=$cfg --data_svc_ip=${ip} --data_svc_port=${data_svc_base_port}
