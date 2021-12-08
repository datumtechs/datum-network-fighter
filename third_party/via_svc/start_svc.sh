
if [ "$1" == "" ]; then
  echo "USAGE:  $0 port" >&2
  exit 1
fi

port=$1
use_ssl=${2:-0}   # 0: not use ssl, 1: use ssl
mkdir -p log
if [ $use_ssl -eq 1 ]
then
  nohup ./via-go -ssl ./conf/ssl-conf.yml -address 0.0.0.0:${port} > log/via_svc_${port}.log 2>&1 &
else
  nohup ./via-go-no_ssl -address 0.0.0.0:${port} > log/via_svc_${port}.log 2>&1 &
fi
