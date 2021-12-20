is_restart_task=${1:-1}  # default not to restart task, set 0 to restart task.

export PYTHONPATH=$PYTHONPATH:..:../protos:../common
mkdir -p log
scripts_path=$(cd $(dirname $0); pwd)
base_dir=${scripts_path}"/../.."
nodes_conf=${scripts_path}/nodes_conf
cfg=config.yaml
ip=$(cat $nodes_conf/config.yaml | shyaml get-value ip.0)
data_svc_base_port=$(cat $nodes_conf/config.yaml | shyaml get-value data_svc_port)

case $is_restart_task in
    0)
        echo "***** restart data&compute&schedule service."
        bash kill.sh
        cd $base_dir
        python3 tools/onekey_deploy.py --start_all $nodes_conf/nodes_conf.json
        ;;
esac

cd $base_dir/console
echo "start console that connect to data_svc which internal port ${data_svc_base_port}"
echo "run task command:  comp_run_task <task_id> <task_cfg_file>"
echo "for example: comp_run_task abc task_cfg_lr_train_cluster.json"
python3 main.py --config=$cfg --data_svc_ip=${ip} --data_svc_port=${data_svc_base_port}
