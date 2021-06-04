import argparse
import logging
import os

import grpc
import pandas as pd
from google.protobuf import empty_pb2
from prompt_toolkit import prompt

from protos import compute_svc_pb2, compute_svc_pb2_grpc
from protos import data_svc_pb2, data_svc_pb2_grpc
from protos import net_comm_svc_pb2_grpc
from protos import schedule_svc_pb2_grpc
from protos import via_svc_pb2

task_id = None


def get_status(args, stub):
    response, call = stub.GetStatus.with_call(
        empty_pb2.Empty(),
        metadata=(
            ('task_id', task_id),
        ))
    print("received: " + response.node_type)


def upload(args, stub):
    if not args:
        print('what do you want to upload?')
        return
    path = args[0]
    if not os.path.exists(path):
        print('path is wrong.')
    response = stub.UploadData(_upload_chunk(path))
    print("received: " + response.data_id)


def upload_dir(args, stub):
    if not args:
        print('where your files?')
        return
    path = args[0]
    if not os.path.exists(path):
        print('path is wrong.')

    def files(dir_):
        for f in os.listdir(dir_):
            file = os.path.join(dir_, f)
            if os.path.isfile(file):
                print(f'upload: {file}')
                for i in _upload_chunk(file):
                    yield i

    response_it = stub.BatchUpload(files(path))
    for ans in response_it:
        print("received: " + ans.data_id)


def _upload_chunk(path):
    print('start sending content')
    with open(path, 'rb') as content_file:
        chunk_size = 1024
        chunk = content_file.read(chunk_size)
        while chunk:
            yield data_svc_pb2.UploadRequest(content=chunk)
            chunk = content_file.read(chunk_size)
    print('sending content done')

    df = pd.read_csv(path)
    meta = data_svc_pb2.FileInfo()
    meta.file_name = os.path.basename(path)
    meta.file_type = os.path.splitext(path)[1]
    cols, dtypes = [], []
    for c, t in df.dtypes.items():
        cols.append(c)
        dtypes.append(t)
    meta.columns.extend(cols)
    meta.col_dtypes.extend(str(dtypes))
    yield data_svc_pb2.UploadRequest(meta=meta)
    print('send meta data done.')


def download(args, stub):
    if not args or len(args) != 2:
        print('Syntax: download data_id save_to')
        return
    data_id = args[0]
    save_to = args[1]
    if os.path.exists(save_to):
        print('file existed')
        return
    with open(save_to, 'w') as fp:
        pass
    f = open(save_to, 'r+b')
    print(f'save to: {save_to}')

    ok = False
    try:
        response_it = stub.DownloadData(data_svc_pb2.DownloadRequest(data_id=data_id))
        for ans in response_it:
            if ans.WhichOneof('data') == 'status':
                print(data_svc_pb2.TaskStatus.Name(ans.status))
                if ans.status != data_svc_pb2.TaskStatus.Finished:
                    os.remove(save_to)
                if ans.status == data_svc_pb2.TaskStatus.Finished:
                    ok = True
                break
            f.write(ans.content)
    except Exception as e:
        print(e)
    finally:
        if f:
            f.close()

    if ok:
        print(f'download file size: {os.path.getsize(save_to)}')


def publish_data():
    pass


def publish_compute_resource():
    pass


def publish_algo():
    pass


def call_contract():
    pass


def list_data(args, stub):
    response = stub.ListData(empty_pb2.Empty())
    for row in response.data:
        print(row)


def shares(args, stub):
    if not args or len(args) != 4:
        print('Syntax: shares data_id p0 p1 p2')
        return
    data_id = args[0]
    parts = args[1:]
    ans = stub.SendSharesData(data_svc_pb2.SendSharesDataRequest(data_id=data_id, receivers=parts))
    print(data_svc_pb2.TaskStatus.Name(ans.status))


def comp_get_status(args, stub):
    resp = stub.GetStatus(empty_pb2.Empty())
    print(resp)


def comp_task_details(args, stub):
    req = compute_svc_pb2.GetTaskDetailsReq()
    req.task_ids.extend(args)
    resp = stub.GetTaskDetails(req)
    print(resp)


def comp_upload_shard(args, stub):
    pass


def comp_run_task(args, stub):
    req = compute_svc_pb2.TaskReadyGoReq()
    req.task_id = args[0]
    req.contract_id = 'C0DE-01'
    req.data_id = 'DA7A-01'
    req.env_id = 'E5'
    req.party_id = 0

    p = req.peers.add()
    p.ip = '192.168.16.151'
    p.port = 50021
    p.party = 0
    p.name = 'P0'

    p = req.peers.add()
    p.ip = '192.168.16.151'
    p.port = 50022
    p.party = 1
    p.name = 'P1'

    p = req.peers.add()
    p.ip = '192.168.16.151'
    p.port = 50023
    p.party = 2
    p.name = 'P2'

    resp = stub.HandleTaskReadyGo(req)
    print(resp)


directions = {
    'status': get_status,
    'list_data': list_data,
    'upload': upload,
    'upload_dir': upload_dir,
    'download': download,
    'shares': shares,
    'pub_data': publish_data,

    'comp_status': comp_get_status,
    'comp_task_details': comp_task_details,
    'comp_upload_shard': comp_upload_shard,
    'comp_run_task': comp_run_task,
}

svc_stub = {via_svc_pb2.DATA_SVC: data_svc_pb2_grpc.DataProviderStub,
            via_svc_pb2.COMPUTE_SVC: compute_svc_pb2_grpc.ComputeProviderStub,
            via_svc_pb2.SCHEDULE_SVC: schedule_svc_pb2_grpc.ScheduleProviderStub,
            via_svc_pb2.NET_COMM_SVC: net_comm_svc_pb2_grpc.NetCommProviderStub
            }

channels = {}
stubs = {}

if __name__ == '__main__':
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_svc_ip', type=str, default='192.168.16.151')
    parser.add_argument('--data_svc_port', type=int, default=50011)
    parser.add_argument('--compute_svc_ip', type=str, default='192.168.16.151')
    parser.add_argument('--compute_svc_port', type=int, default=50021)
    parser.add_argument('--task_id', type=str)
    args = parser.parse_args()

    task_id = args.task_id
    addr = f'{args.data_svc_ip}:{args.data_svc_port}'
    ch = grpc.insecure_channel(addr)
    stubs[via_svc_pb2.DATA_SVC] = svc_stub[via_svc_pb2.DATA_SVC](ch)
    channels[via_svc_pb2.DATA_SVC] = ch
    addr = f'{args.compute_svc_ip}:{args.compute_svc_port}'
    ch = grpc.insecure_channel(addr)
    stubs[via_svc_pb2.COMPUTE_SVC] = svc_stub[via_svc_pb2.COMPUTE_SVC](ch)
    channels[via_svc_pb2.COMPUTE_SVC] = ch

    while True:
        user_input = prompt('> ')
        if user_input == 'exit':
            break
        user_input = user_input.strip().split()
        if not user_input:
            continue
        cmd = user_input[0]
        if cmd not in directions:
            continue
        if cmd.startswith('comp_'):
            svc_type = via_svc_pb2.COMPUTE_SVC
        else:
            svc_type = via_svc_pb2.DATA_SVC
        stub = stubs[svc_type]
        directions[cmd](user_input[1:], stub)

    stubs.clear()
    for ch in channels:
        ch.close()
