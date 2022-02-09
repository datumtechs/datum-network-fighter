
bash kill.sh

source config.ini
via_svc_num=${via_svc_num}
schedule_svc_num=$[${data_svc_num} + ${compute_svc_num}]
schedule_port=${schedule_svc_base_port}

export PYTHONPATH=$PYTHONPATH:..:../lib:../common
mkdir -p log
scripts_path=$(cd $(dirname $0); pwd)
log_path=${scripts_path}"/log"
base_dir=${scripts_path}"/../.."
cfg=config.yaml
ip=127.0.0.1
use_ssl=0      # 0: not use ssl,  1: use ssl
use_consul=1   # 0: not use consul, 1: use consul
# if modify, must absolute path to python37
python_command=python3

############## consul_svc #############
if [ $use_consul -ne 0 ]
then
    cd $base_dir/third_party/consul_server
    bash run_consul.sh ${ip}
    sleep 5
fi

############## data_svc #############
cd $base_dir/data_svc
data_svc_log=${log_path}"/data_svc"
mkdir -p ${data_svc_log}
for port in $(seq ${data_svc_base_port} $[${data_svc_base_port}+${data_svc_num}-1])
do 
    echo "start data_svc that use port ${port}"
    nohup $python_command main.py $cfg --bind_ip=${ip} --port=${port} --schedule_svc=${ip}:${schedule_port} --use_consul=${use_consul} > ${data_svc_log}/data_svc_${port}.log 2>&1 &
    if [ $use_consul -eq 0 ]
    then
        schedule_port=$[${schedule_port}+1]
    fi
done


############## compute_svc #############
cd $base_dir/compute_svc
compute_svc_log=${log_path}"/compute_svc"
mkdir -p ${compute_svc_log}
for port in $(seq ${compute_svc_base_port} $[${compute_svc_base_port}+${compute_svc_num}-1])
do 
    echo "start compute_svc that use port ${port}"
    nohup $python_command main.py $cfg --bind_ip=${ip} --port=${port} --schedule_svc=${ip}:${schedule_port} --use_consul=${use_consul} > ${compute_svc_log}/compute_svc_${port}.log 2>&1 &
    if [ $use_consul -eq 0 ]
    then
        schedule_port=$[${schedule_port}+1]
    fi
done


############## via_svc #############
cd $base_dir/third_party/via_svc
via_svc_log=${log_path}"/via_svc"
mkdir -p ${via_svc_log}
for port in $(seq ${via_svc_base_port} $[${via_svc_base_port}+${via_svc_num}-1])
do
    echo "start via_svc that use port ${port}"
    if [ $use_ssl -eq 1 ]
    then
        nohup ./via-go -ssl ./conf/ssl-conf.yml -address 0.0.0.0:${port} > ${via_svc_log}/via_svc_${port}.log 2>&1 &
    else
        nohup ./via-go-no_ssl -address 0.0.0.0:${port} > ${via_svc_log}/via_svc_${port}.log 2>&1 &
    fi
done


############## schedule_svc #############
cd $base_dir/tests/schedule_svc
schedule_svc_log=${log_path}"/schedule_svc"
mkdir -p ${schedule_svc_log}
if [ $use_consul -eq 0 ]
then
    for port in $(seq ${schedule_svc_base_port} $[${schedule_svc_base_port}+${schedule_svc_num}-1])
    do
        echo "start schedule_svc that use port ${port}"
        nohup $python_command main.py $cfg --bind_ip=${ip} --port=${port} --use_consul=${use_consul} > ${schedule_svc_log}/schedule_svc_${port}.log 2>&1 &
    done
else
    nohup $python_command main.py $cfg --bind_ip=${ip} --port=${schedule_port} --use_consul=${use_consul} > ${schedule_svc_log}/schedule_svc_${schedule_port}.log 2>&1 &
fi


############## console #############
cd $base_dir/console
echo "start console that connect to data_svc which internal port ${data_svc_base_port}"
echo "run task command:  comp_run_task <task_id> <task_cfg_file>"
echo "for example:  comp_run_task abc task_cfg_lr_train.json"
$python_command main.py --config=$cfg --data_svc_ip=${ip} --data_svc_port=${data_svc_base_port}
