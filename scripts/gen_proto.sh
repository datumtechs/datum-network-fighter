
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./data_svc.proto

PYTHONPATH="..:../protos" python main.py
