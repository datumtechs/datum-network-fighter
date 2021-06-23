#!/bin/bash
function stop_gateway(){
    status=$1
    node_name_list=`ps aux |grep gateway|grep -v start_gateway.sh|grep -v grep |awk {'print $2'}`;
    echo "Your status is $status"
    if [ -z "$node_name_list" ];then
       if [ $status == "stop" ];then
	        exit 0
       else
	        return
       fi
    fi

    for each in $node_name_list
        do
	          kill $each
        done
    if [ $status == "stop" ];then
       exit 0
    fi
}
stop_gateway $1

if [ ! -d compute_svc  ];then
  mkdir compute_svc
fi
if [ ! -d data_svc  ];then
  mkdir data_svc
fi
Filghter_Path=../armada-common/Fighter
cp generate_gateway.sh $Filghter_Path
cd $Filghter_Path && ./generate_gateway.sh && cd -
#stop_gateway
go run compute_gateway.go 2>&1  &
go run data_gateway.go 2>&1  &
