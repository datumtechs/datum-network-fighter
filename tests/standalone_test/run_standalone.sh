
bash kill.sh

source config.ini
via_svc_num=${via_svc_num}
schedule_svc_num=$[${data_svc_num} + ${compute_svc_num}]
schedule_port=${schedule_svc_base_port}

export PYTHONPATH=$PYTHONPATH:..:../protos:../common
mkdir -p log
scripts_path=$(cd $(dirname $0); pwd)
log_path=${scripts_path}"/log"
base_dir=${scripts_path}"/../.."
cfg=config.yaml
ip=127.0.0.1
# if modify, must absolute path to python37
python_command=python3


############## data_svc #############
cd $base_dir/data_svc
data_svc_log=${log_path}"/data_svc"
mkdir -p ${data_svc_log}
for port in $(seq ${data_svc_base_port} $[${data_svc_base_port}+${data_svc_num}-1])
do 
    echo "start data_svc that use port ${port}"
    nohup $python_command main.py $cfg --bind_ip=${ip} --port=${port} --schedule_svc=${ip}:${schedule_port} > ${data_svc_log}/data_svc_${port}.log 2>&1 &
    schedule_port=$[${schedule_port}+1]
done


############## compute_svc #############
cd $base_dir/compute_svc
compute_svc_log=${log_path}"/compute_svc"
mkdir -p ${compute_svc_log}
for port in $(seq ${compute_svc_base_port} $[${compute_svc_base_port}+${compute_svc_num}-1])
do 
    echo "start compute_svc that use port ${port}"
    nohup $python_command main.py $cfg --bind_ip=${ip} --port=${port} --schedule_svc=${ip}:${schedule_port} > ${compute_svc_log}/compute_svc_${port}.log 2>&1 &
    schedule_port=$[${schedule_port}+1]
done


############## via_svc #############
cd $base_dir/third_party/via_svc
via_svc_log=${log_path}"/via_svc"
mkdir -p ${via_svc_log}
for port in $(seq ${via_svc_base_port} $[${via_svc_base_port}+${via_svc_num}-1])
do
    echo "start via_svc that use port ${port}"
    nohup ./via-go -ssl ./conf/ssl-conf.yml -address 0.0.0.0:${port} > ${via_svc_log}/via_svc_${port}.log 2>&1 &
done


############## schedule_svc #############
cd $base_dir/tests/schedule_svc
schedule_svc_log=${log_path}"/schedule_svc"
mkdir -p ${schedule_svc_log}
for port in $(seq ${schedule_svc_base_port} $[${schedule_svc_base_port}+${schedule_svc_num}-1])
do
    echo "start schedule_svc that use port ${port}"
    PYTHONPATH="../..:../../protos/:../../common" nohup $python_command main.py $cfg --bind_ip=${ip} --port=${port} > ${schedule_svc_log}/schedule_svc_${port}.log 2>&1 &
done


############## console #############
cd $base_dir/console
echo "start console that connect to data_svc which internal port ${data_svc_base_port}"
echo "run task command:  comp_run_task <task_id> <task_cfg_file>"
$python_command main.py --config=$cfg --data_svc_ip=${ip} --data_svc_port=${data_svc_base_port}
