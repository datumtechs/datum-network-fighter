rm -rf data-dir/*
rm -rf config-dir/*
bind_ip=$1
port=8500
if [[ -n $(lsof -i:$port | awk '{print $2}' | grep -v PID) ]]
then
    echo "kill consul process that use port ${port}"
    lsof -i:$port | awk '{print $2}' | grep -v PID | xargs kill
fi
echo "start consul server that use address ${bind_ip}:${port}"
nohup ./consul agent -server -bootstrap-expect 1 -data-dir ./data-dir -node=jatel -bind=${bind_ip} -ui -rejoin -config-dir=./config-dir -client 0.0.0.0 > consul_server.log  2>&1 &
