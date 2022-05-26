#!/bin/bash
python_interpreter=$3
if [ "$1" = "" ]; then
  echo "USAGE:  $0 config" >&2
  exit 1
fi

cfg=$1
# shellcheck disable=SC2206
deploy_dir=(${0//$2/ })
# shellcheck disable=SC2128
log_dir=$deploy_dir$2/log
if [ ! -e "$log_dir" ];then
    # shellcheck disable=SC2086
    mkdir -p $log_dir
fi

flagData="data"
flagCompute="compute"
isData=$(echo "$2" | grep "${flagData}")
isCompute=$(echo "$2" | grep "${flagCompute}")
if [[ "$isData" != "" ]];then
  echo start data services
  $python_interpreter -u -m fighter.data_svc.main "$cfg" >> "$log_dir"/data.log 2>&1
fi

if [[ "$isCompute" != "" ]];then
  echo start compute services
  $python_interpreter -u -m fighter.compute_svc.main "$cfg" >> "$log_dir"/compute.log 2>&1
fi