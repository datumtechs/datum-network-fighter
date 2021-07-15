
if [ "$1" == "" ]; then
  echo "USAGE:  $0 config" >&2
  exit 1
fi

cfg=$1

PYTHONPATH="..:../protos/:../common" python main.py --config $cfg
