
if [ "$1" == "" ]; then
  echo "USAGE:  $0 config" >&2
  exit 1
fi

cfg=$1
log=${cfg/yaml/log}
nohup python -u main.py $cfg >$log 2>&1 &
