
source config.ini
via_svc_num=$[${data_svc_num} + ${compute_svc_num}]
schedule_svc_num=$[${data_svc_num} + ${compute_svc_num}]


############## data_svc #############
for port in $(seq ${data_svc_base_port} $[${data_svc_base_port}+${data_svc_num}-1])
do
    if [[ -n $(lsof -i:$port | awk '{print $2}' | grep -v PID) ]]
    then
        echo "kill data_svc process that use port ${port}"
        lsof -i:$port | awk '{print $2}' | grep -v PID | xargs kill
    fi
done


############## compute_svc #############
for port in $(seq ${compute_svc_base_port} $[${compute_svc_base_port}+${compute_svc_num}-1])
do
    if [[ -n $(lsof -i:$port | awk '{print $2}' | grep -v PID) ]]
    then
        echo "kill compute_svc process that use port ${port}"
        lsof -i:$port | awk '{print $2}' | grep -v PID | xargs kill
    fi
done


############## via_svc #############
for port in $(seq ${via_svc_base_port} $[${via_svc_base_port}+${via_svc_num}-1])
do
    if [[ -n $(lsof -i:$port | awk '{print $2}' | grep -v PID) ]]
    then
        echo "kill via_svc process that use port ${port}"
        lsof -i:$port | awk '{print $2}' | grep -v PID | xargs kill
    fi
done


############## schedule_svc #############
for port in $(seq ${schedule_svc_base_port} $[${schedule_svc_base_port}+${schedule_svc_num}-1])
do
    if [[ -n $(lsof -i:$port | awk '{print $2}' | grep -v PID) ]]
    then
        echo "kill schedule_svc process that use port ${port}"
        lsof -i:$port | awk '{print $2}' | grep -v PID | xargs kill -9
    fi
done

# if [[ -n $(ps -aux | grep python | grep -v grep | awk '{print $2}') ]]
# then
#     ps -aux | grep python | grep -v grep | awk '{print $2}' | xargs sudo kill -9
# fi
