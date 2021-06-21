#!/bin/bash
function stop_gateway(){
    node_name_list=`ps aux |grep gateway|grep -v grep |awk {'print $2'}`;
    for each in $node_name_list
        do
	          kill $each
        done
}
if [ ! -d compute_svc  ];then
  mkdir compute_svc
fi
if [ ! -d data_svc  ];then
  mkdir data_svc
fi

cd ../protos/ && ./generate_gateway.sh && cd -
stop_gateway
go run compute_gateway.go 2>&1  &
go run data_gateway.go 2>&1  &
