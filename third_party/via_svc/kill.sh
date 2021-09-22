for port in $(seq 50041 1 50049)
do
    if [[ -n $(lsof -i:$port | awk '{print $2}' | grep -v PID) ]]
    then
        echo "kill via process that use port ${port}"
        lsof -i:$port | awk '{print $2}' | grep -v PID | xargs kill -9
    fi
done
