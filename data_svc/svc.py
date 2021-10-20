import hashlib
import logging
import os
import threading
import time
import math
import psutil
import grpc

from common.report_engine import report_upload_file_summary
from config import cfg
from protos import common_pb2
from protos import compute_svc_pb2, compute_svc_pb2_grpc
from protos import data_svc_pb2, data_svc_pb2_grpc


log = logging.getLogger(__name__)

def get_sys_stat(cfg):
    stat = data_svc_pb2.GetStatusReply()
    stat.total_cpu = psutil.cpu_count()
    stat.used_cpu = math.ceil(stat.total_cpu * psutil.cpu_percent(0.1) / 100)
    stat.idle_cpu = max(stat.total_cpu - stat.used_cpu, 0)
    
    stat.total_memory = psutil.virtual_memory().total
    stat.used_memory = psutil.virtual_memory().used
    stat.idle_memory = psutil.virtual_memory().free
    
    stat.total_disk = psutil.disk_usage('/').total
    stat.used_disk = psutil.disk_usage('/').used
    stat.idle_disk = psutil.disk_usage('/').free
    
    stat.total_bandwidth = cfg["total_bandwidth"]
    net_1 = psutil.net_io_counters()
    time.sleep(1)
    net_2 = psutil.net_io_counters()
    stat.used_bandwidth = (net_2.bytes_sent - net_1.bytes_sent) + (net_2.bytes_recv - net_1.bytes_recv)
    stat.idle_bandwidth = max(stat.total_bandwidth - stat.used_bandwidth, 0)
    str_res = '{' + str(stat).replace('\n', ' ').replace('  ', ' ').replace('{', ':{') + '}'
    log.info(f"get sys stat: {str_res}")
    return stat


class DataProvider(data_svc_pb2_grpc.DataProviderServicer):
    def __init__(self, task_manager):
        self.task_manager = task_manager
        print(f'cur thread id: {threading.get_ident()}')

    def GetStatus(self, request, context):
        return get_sys_stat(self.task_manager.cfg)
    
    def UploadData(self, request_it, context):
        print(type(request_it))
        folder = cfg['data_root']
        folder = os.path.abspath(folder)
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
                    m.update(new_name.encode())
                    full_new_name = os.path.join(folder, new_name)
                    print(full_new_name)
                    os.rename(path, full_new_name)
                    data_id = m.hexdigest()
                    result = data_svc_pb2.UploadReply(ok=True, data_id=data_id, file_path=full_new_name)
                    file_summary = {"origin_id": data_id, "file_path": full_new_name, "ip": cfg["bind_ip"],
                                    "port": cfg["port"]}
                    ret = report_upload_file_summary(cfg['schedule_svc'], file_summary)
                    log.info(ret)
                    if ret and ret.status == 0:
                        return result
                    else:
                        return data_svc_pb2.UploadReply(ok=False)
                else:
                    f.write(req.content)
                    m.update(req.content)
        except Exception as e:
            log.error(repr(e))
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
                result = data_svc_pb2.UploadReply(ok=True, data_id=data_id, file_path=full_name)
                yield result
            else:
                f.write(req.content)
                m.update(req.content)

    def DownloadData(self, request, context):
        try:
            folder = cfg['data_root']
            path = os.path.join(folder, os.path.basename(request.file_path))
            log.info(f'to download: {path}')
            if not os.path.exists(path):
                yield data_svc_pb2.DownloadReply(status=data_svc_pb2.TaskStatus.Failed)
            else:
                log.info('start sending content')
                with open(path, 'rb') as content_file:
                    chunk_size = cfg['chunk_size']
                    chunk = content_file.read(chunk_size)
                    while chunk:
                        yield data_svc_pb2.DownloadReply(content=chunk)
                        chunk = content_file.read(chunk_size)
                yield data_svc_pb2.DownloadReply(status=data_svc_pb2.TaskStatus.Finished)
                log.info('sending content done')
        except Exception as e:
            log.error(repr(e))

    def SendSharesData(self, request, context):
        ans = data_svc_pb2.SendSharesDataReply(status=data_svc_pb2.TaskStatus.Cancelled)
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
        log.info(f'{context.peer()} submit a task {request.task_id}, thread id: {threading.get_ident()}')
        ok, msg = self.task_manager.start(request)
        return common_pb2.TaskReadyGoReply(ok=ok, msg=msg)

    def HandleCancelTask(self, request, context):
        log.info(f'{context.peer()} want to cancel task {request.task_id}')
        ok, msg = self.task_manager.cancel_task(request.task_id)
        return common_pb2.TaskCancelReply(ok=ok, msg=msg)
