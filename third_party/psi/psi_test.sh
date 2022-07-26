echo -e "id\nzhangsan3\nlisi4\nwangwu5\n" > a.csv && echo -e "id\nwangwu5\nlisi4\n" > b.csv
cp ../../doc/ioconfig.json .
python3 psi_demo.py --node_id=P0 --ioconfig=ioconfig.json --psi_type=T_V1_Basic_GLS254 --input=a.csv --output=a_result > test_a.log 2>&1 &
python3 psi_demo.py --node_id=P1 --ioconfig=ioconfig.json --psi_type=T_V1_Basic_GLS254 --input=b.csv --output=b_result
echo "---------------"
cat b_result.taskid-0.csv
