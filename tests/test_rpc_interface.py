import os
import grpc
import unittest
import pandas as pd
import configparser
import time
from google.protobuf import empty_pb2
from protos import compute_svc_pb2, compute_svc_pb2_grpc
from protos import data_svc_pb2, data_svc_pb2_grpc
from protos import common_pb2

config = configparser.ConfigParser()
config.read('../gateway/common/config.ini')
task_id = '123456'


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


def get_status(stub):
    try:
        response, call = stub.GetStatus.with_call(
            empty_pb2.Empty(),
            metadata=(
                ('task_id', task_id),
            ))
        print("received: " + response.node_type)
        return True
    except Exception as e:
        print(e)
        return False


def upload(stub, file_path):
    try:
        if not os.path.exists(file_path):
            print('path is wrong.')
            return False
        response = stub.UploadData(_upload_chunk(file_path))
        print("received: " + response.data_id)
        return True
    except Exception as e:
        print(e)
        return False


def upload_dir(stub, file_dir):
    try:
        if not os.path.exists(file_dir):
            print(f'file_dir {file_dir} is wrong.')
            return False

        def files(dir_):
            for f in os.listdir(dir_):
                file = os.path.join(dir_, f)
                if os.path.isfile(file):
                    print(f'upload: {file}')
                    yield from _upload_chunk(file)

        response_it = stub.BatchUpload(files(file_dir))
        for ans in response_it:
            print("received: " + ans.data_id)
        return True
    except Exception as e:
        print(e)
        return False


def download(stub, file_path, file_name):
    if os.path.exists(file_path):
        print('file existed')
        return False
    with open(file_path, 'w') as fp:
        pass
    f = open(file_path, 'r+b')
    print(f'save to: {file_path}')

    ok = False
    try:
        response_it = stub.DownloadData(data_svc_pb2.DownloadRequest(file_path=file_name))
        for ans in response_it:
            if ans.WhichOneof('data') == 'status':
                print(data_svc_pb2.TaskStatus.Name(ans.status))
                if ans.status != data_svc_pb2.TaskStatus.Finished:
                    os.remove(file_path)
                if ans.status == data_svc_pb2.TaskStatus.Finished:
                    ok = True
                break
            f.write(ans.content)
    except Exception as e:
        print(e)
        return False
    finally:
        if f:
            f.close()

    if ok:
        print(f'download file size: {os.path.getsize(file_path)}')
    return True


def list_data(stub):
    try:
        response = stub.ListData(empty_pb2.Empty())
        for row in response.data:
            print(row)
        return True
    except Exception as e:
        print(e)
        return False


def comp_get_status(stub):
    try:
        resp = stub.GetStatus(empty_pb2.Empty())
        print(resp)
        return True
    except Exception as e:
        print(e)
        return False


def comp_task_details(args, stub):
    try:
        req = compute_svc_pb2.GetTaskDetailsReq()
        req.task_ids.extend(args)
        resp = stub.GetTaskDetails(req)
        print(resp)
        return True
    except Exception as e:
        print(e)
        return False


def comp_run_task(args, stub):
    try:
        req = common_pb2.TaskReadyGoReq()
        req.task_id = task_id
        req.contract_id = args['contract_id']
        req.data_id = args['data_id']
        req.env_id = args['env_id']
        req.party_id = args['party_id']
        for peer in args['peers']:
            p = req.peers.add()
            p.ip = peer['ip']
            p.port = peer['port']
            p.party = peer['party']
            p.name = peer['name']
        resp = stub.HandleTaskReadyGo(req)
        print(resp)
        return True
    except Exception as e:
        print(e)
        return False


class UnitTest(unittest.TestCase):
    addr = f"{config.get('DataSvc', 'rpc_host')}:{config.get('DataSvc', 'rpc_port')}"
    ch = grpc.insecure_channel(addr)
    data_stub = data_svc_pb2_grpc.DataProviderStub(ch)

    addr = f"{config.get('ComputeSvc', 'rpc_host')}:{config.get('ComputeSvc', 'rpc_port')}"
    ch = grpc.insecure_channel(addr)
    comp_stub = compute_svc_pb2_grpc.ComputeProviderStub(ch)

    def test_get_status(self):
        self.assertEqual(True, get_status(self.data_stub))

    def test_upload(self):
        self.assertEqual(True, upload(self.data_stub, './test_data/p1.csv'))

    def test_upload_dir(self):
        self.assertEqual(True, upload_dir(self.data_stub, './test_data/'))

    def test_download(self):
        now = time.strftime("%Y%m%d-%H%M%S")
        self.assertEqual(True, download(self.data_stub, f'./test_data/download_{now}.csv', 'p0_20210623-124735.csv'))

    def test_list_data(self):
        self.assertEqual(True, list_data(self.data_stub))

    def test_comp_get_status(self):
        self.assertEqual(True, comp_get_status(self.comp_stub))

    def test_comp_task_details(self):
        self.assertEqual(True, comp_task_details(['1'], self.comp_stub))

    def test_comp_run_task(self):
        args = {
            'contract_id': '7654321',
            'data_id': '7777777777',
            'party_id': 1,
            'env_id': '9999999999',
            'peers': [{'ip': '192.168.235.151', 'port': 4565, 'party': 0, 'name': 'Tom'},
                      {'ip': '192.168.235.151', 'port': 4567, 'party': 1, 'name': 'Jerry'},
                      {'ip': '192.168.235.151', 'port': 4568, 'party': 2, 'name': 'Peter'},
                      ]
        }
        self.assertEqual(True, comp_run_task(args, self.comp_stub))


unittest.main(verbosity=1)
