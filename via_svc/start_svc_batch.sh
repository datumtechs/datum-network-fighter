if [ "$1" == "" ]; then
  echo "USAGE:  $0 NUM_PORTS_TO_OPEN FROM" >&2
  exit 1
fi

n_ports=$1
port_from=$2

for i in $( seq 1 $n_ports )
do
    port=$(( $port_from + $i ))
    PYTHONPATH="..:../protos" python main.py --port $port &
done    
