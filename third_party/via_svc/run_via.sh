
bash kill.sh

mkdir -p log
for port in $(seq 50041 1 50049)
do
    echo "start via_svc that use port ${port}"
    nohup ./via-go -ssl ./conf/ssl-conf.yml -address 0.0.0.0:${port} > log/via_${port}.log 2>&1 &
done
