#!/bin/bash
python_interpreter=$3
if [ "$1" = "" ]; then
  echo "USAGE:  $0 config" >&2
  exit 1
fi

cfg=$1
log_dir="$HOME"/data/yaml
if [ ! -e "$log_dir" ];then
    # shellcheck disable=SC2086
    mkdir -p $log_dir
fi

if [ "$2" = "data" ];then
  echo start data services
  $python_interpreter -u -m metis.data_svc.main "$cfg" >"$log_dir"/log 2>&1
fi

if [ "$2" = "compute" ];then
  echo start compute services
  $python_interpreter -u -m metis.compute_svc.main "$cfg" >"$log_dir"/log 2>&1
fi