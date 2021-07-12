import hashlib
import os
import threading
import time

import grpc

from config import cfg
from protos import compute_svc_pb2, compute_svc_pb2_grpc
from protos import data_svc_pb2, data_svc_pb2_grpc
from protos import common_pb2
from third_party.rosetta_helper import split_data


class DataProvider(data_svc_pb2_grpc.DataProviderServicer):
    def __init__(self):
        print(f'cur thread id: {threading.get_ident()}')

    def GetStatus(self, request, context):
        print(context.peer(), f'cur thread id: {threading.get_ident()}')
        return data_svc_pb2.GetStatusReply(node_type='data_node')

    def UploadData(self, request_it, context):
        print(type(request_it))
        folder = cfg['data_root']
        now = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, now)
        print(f'save to {path}')
        m = hashlib.sha256()
        with open(path, 'w') as fp:
            pass
        f = open(path, 'r+b')

        try:
            for req in request_it:
                if req.WhichOneof('data') == 'meta':
                    file = req.meta.file_name
                    cols = req.meta.columns
                    print(file, len(cols), ','.join(cols))

                    stem, ext = os.path.splitext(os.path.basename(file))
                    new_name = f'{stem}_{now}{ext}'
                    full_new_name = os.path.join(folder, new_name)
                    print(full_new_name)
                    os.rename(path, full_new_name)
                    data_id = m.hexdigest()
                    result = data_svc_pb2.UploadReply(ok=True, data_id=data_id, file_path=new_name)
                    return result
                else:
                    f.write(req.content)
                    m.update(req.content)
                    data_svc_pb2.UploadReply(ok=False)
        finally:
            if f:
                f.close()

    def BatchUpload(self, request_it, context):
        print(context.peer(), f'cur thread id: {threading.get_ident()}')
        folder = cfg['data_root']

        state = 0
        for i, req in enumerate(request_it):
            if state == 0:
                now = time.strftime("%Y%m%d-%H%M%S")
                path = os.path.join(folder, now)
                print(f'save to {path}')
                m = hashlib.sha256()
                with open(path, 'w') as fp:
                    pass
                f = open(path, 'r+b')
                state = 1

            if req.WhichOneof('data') == 'meta':
                file = req.meta.file_name
                cols = req.meta.columns
                print(file, len(cols), ','.join(cols))

                stem, ext = os.path.splitext(os.path.basename(file))
                new_name = f'{stem}_{now}{ext}'
                full_name = os.path.join(folder, new_name)
                print(full_name)
                os.rename(path, full_name)
                data_id = m.hexdigest()
                state = 0
                if f:
                    f.close()
                result = data_svc_pb2.UploadReply(ok=True, data_id=data_id, file_path=new_name)
                yield result
            else:
                f.write(req.content)
                m.update(req.content)
                data_svc_pb2.UploadReply(ok=False)

    def DownloadData(self, request, context):
        folder = cfg['data_root']
        path = os.path.join(folder, os.path.basename(request.data_id))
        print(f'to download: {path}')
        if not os.path.exists(path):
            yield data_svc_pb2.DownloadReply(status=data_svc_pb2.TaskStatus.Failed)
        else:
            print('start sending content')
            with open(path, 'rb') as content_file:
                chunk_size = cfg['chunk_size']
                chunk = content_file.read(chunk_size)
                while chunk:
                    yield data_svc_pb2.DownloadReply(content=chunk)
                    chunk = content_file.read(chunk_size)
                yield data_svc_pb2.DownloadReply(status=data_svc_pb2.TaskStatus.Finished)
            print('sending content done')

    def SendSharesData(self, request, context):
        n_parts = len(request.receivers)
        parts = split_data(request.data_id, n_parts)
        for dat, addr in zip(parts, request.receivers):
            self._send_dat(dat, addr)
        ans = data_svc_pb2.SendSharesDataReply(status=data_svc_pb2.TaskStatus.Start)
        return ans

    def _send_dat(self, dat, to):
        print(f'async send data(size: {dat.size}) to {to}')
        with grpc.insecure_channel(to) as channel:
            stub = compute_svc_pb2_grpc.ComputeProviderStub(channel)
            req = compute_svc_pb2.UploadShardReq(task_id='0xC0DE01', data_id='0xDA7A1234')
            response = stub.UploadShard(req)
            print(response.ok)

    def ListData(self, request, context):
        print(context.peer())
        ans = data_svc_pb2.ListDataReply()
        folder = cfg['data_root']
        if not os.path.exists(folder):
            os.makedirs(folder)
        files = os.listdir(folder)
        for f in files:
            path = os.path.join(folder, f)
            sz = os.path.getsize(path)
            print(f'{f}: {sz}')
            row = ans.data.add()
            row.file_name = f
            row.size = sz
        return ans

    def HandleTaskReadyGo(self, request, context):
        return common_pb2.TaskReadyGoReply(ok=False, msg='no impl')
