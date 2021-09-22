
bash kill.sh

workdir=$(cd $(dirname $0); pwd)
workdir=${workdir}"/../.."
cfg=config.yaml
export PYTHONPATH=$PYTHONPATH:..:../protos/:../common
mkdir -p ../log

cd $workdir; cd data_svc
echo "start data_svc that use port 50011,50012,50013"
nohup python -u main.py $cfg --port=50011 --via_svc=192.168.16.151:50041 > ../log/data_svc_50011.log 2>&1 &
nohup python -u main.py $cfg --port=50012 --via_svc=192.168.16.151:50042 > ../log/data_svc_50012.log 2>&1 &
nohup python -u main.py $cfg --port=50013 --via_svc=192.168.16.151:50043 > ../log/data_svc_50013.log 2>&1 &
nohup python -u main.py $cfg --port=50017 --via_svc=192.168.16.151:50047 > ../log/data_svc_50017.log 2>&1 &
nohup python -u main.py $cfg --port=50018 --via_svc=192.168.16.151:50048 > ../log/data_svc_50018.log 2>&1 &
nohup python -u main.py $cfg --port=50019 --via_svc=192.168.16.151:50049 > ../log/data_svc_50019.log 2>&1 &

cd $workdir; cd compute_svc
echo "start compute_svc that use port 50014,50015,50016"
nohup python -u main.py $cfg --port=50014 --via_svc=192.168.16.151:50044 > ../log/comp_svc_50014.log 2>&1 &
nohup python -u main.py $cfg --port=50015 --via_svc=192.168.16.151:50045 > ../log/comp_svc_50015.log 2>&1 &
nohup python -u main.py $cfg --port=50016 --via_svc=192.168.16.151:50046 > ../log/comp_svc_50016.log 2>&1 &

cd $workdir; cd third_party/via_svc
bash run_via.sh

cd $workdir; cd console
python get_config.py
echo "start console that connect to data_svc which use via port 50041"
python -u main.py --config=$cfg --data_svc_ip=192.168.16.151 --data_svc_port=50041
