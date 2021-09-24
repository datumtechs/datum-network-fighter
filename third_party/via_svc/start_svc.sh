
if [ "$1" == "" ]; then
  echo "USAGE:  $0 port" >&2
  exit 1
fi

port=$1
mkdir -p log
nohup ./via-go -ssl ./conf/ssl-conf.yml -address 0.0.0.0:${port} > log/via_svc_${port}.log 2>&1 &