for port in $(seq 50011 1 50019)
do
    if [[ -n $(lsof -i:$port | awk '{print $2}' | grep -v PID) ]]
    then
        echo "kill process that use port ${port}"
        lsof -i:$port | awk '{print $2}' | grep -v PID | xargs kill -9
    fi
done
