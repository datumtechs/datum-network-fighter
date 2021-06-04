#!/bin/bash
if [ ! -d compute_svc  ];then
  mkdir compute_svc
fi
if [ ! -d data_svc  ];then
  mkdir data_svc
fi
cd ../protos/ && ./generate_gateway.sh && cd -
go run compute_gateway.go 2>&1  &
go run data_gateway.go 2>&1  &
