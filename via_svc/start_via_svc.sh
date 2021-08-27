#!/bin/bash

if [ "$1" == "" ]; then
  echo "USAGE:  $0 config" >&2
  exit 1
fi

cfg=$1
log=${cfg/yaml/log}
python -u -m metis.via_svc.main $cfg >$log 2>&1
