import hashlib
import logging
import math
import os
import threading
import time
import json
import grpc
import psutil
try:
    from config import cfg
except:
    from metis.data_svc.config import cfg
from common.report_engine import report_task_result
from common.consts import ERROR_CODE
from lib import common_pb2
from lib import compute_svc_pb2, compute_svc_pb2_grpc
from lib import data_svc_pb2, data_svc_pb2_grpc
from lib.types import base_pb2

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

    stat.status = ERROR_CODE["OK"]
    stat.msg = 'get system status success.'
    str_res = '{' + str(stat).replace('\n', ' ').replace('  ', ' ').replace('{', ':{') + '}'
    log.debug(f"get sys stat: {str_res}")
    return stat


class DataProvider(data_svc_pb2_grpc.DataProviderServicer):
    def __init__(self, task_manager):
        self.task_manager = task_manager
        log.info(f'cur thread id: {threading.get_ident()}')

    def GetStatus(self, request, context):
        return get_sys_stat(self.task_manager.cfg)

    def UploadData(self, request_it, context):
        '''
        message UploadRequest {
            string data_name = 1;
            bytes content = 2;
            string data_type = 3;
            string description = 4;
            repeated string columns = 5;
            repeated string col_dtypes = 6;
            repeated string keywords = 7;
        }
        '''
        log.info(type(request_it))
        folder = cfg['data_root']
        folder = os.path.abspath(folder)
        now = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, now)
        log.info(f'save to {path}')
        m = hashlib.sha256()
        with open(path, 'w') as fp:
            pass
        f = open(path, 'r+b')
        try:
            status = ERROR_CODE["UPLOAD_CONTENT_ERROR"]
            for req in request_it:
                f.write(req.content)
                m.update(req.content)
                data_name = req.data_name
                cols = req.columns
                data_type = req.data_type

            log.info(f"origin filename: {data_name}, len(columns): {len(cols)}, columns:{','.join(cols)}")
            status = ERROR_CODE["GENERATE_SUMMARY_ERROR"]
            data_hash = m.hexdigest()
            stem, ext = os.path.splitext(os.path.basename(data_name))
            new_name = f'{stem}_{now}{ext}'
            m.update(new_name.encode())
            full_new_name = os.path.join(folder, new_name)
            log.info(f'full_new_name: {full_new_name}')
            os.rename(path, full_new_name)
            origin_id = m.hexdigest()

            metadata_option = {"originId": origin_id, "dataPath": full_new_name}
            if data_type == base_pb2.OrigindataType_CSV:
                metadata_option["size"] = os.path.getsize(full_new_name)
                metadata_option["rows"] = 0
                metadata_option["columns"] = 0
                metadata_option["hasTitle"] = True
                metadata_option["metadataColumns"] = []
            elif data_type == base_pb2.OrigindataType_DIR:
                raise NotImplementedError("TO DO UPLOAD DIR.")
            else:
                metadata_option["size"] = os.path.getsize(full_new_name)
            metadata_option = json.dumps(metadata_option)
            file_summary = {"origin_id": origin_id, "ip": cfg["bind_ip"], "port": cfg["port"], "data_hash": data_hash,
                            "data_type": data_type, "metadata_option": metadata_option}
            status = ERROR_CODE["REPORT_SUMMARY_ERROR"]
            ret = report_task_result(cfg['schedule_svc'], 'upload_file', file_summary)
            log.info(f'report_upload_file_summary return: {ret}')
            if ret and ret.status == 0:
                status = ERROR_CODE["OK"]
                result = data_svc_pb2.UploadReply(status=status, msg='upload success.',\
                                data_id=origin_id, data_path=full_new_name, data_hash=data_hash)
            else:
                result = data_svc_pb2.UploadReply(status=status, msg='report summary fail.')
            return result
        except Exception as e:
            log.error(repr(e))
            result = data_svc_pb2.UploadReply(status=status, msg=f'{str(e)[:100]}')
            return result
        finally:
            if f:
                f.close()

    def BatchUpload(self, request_it, context):
        raise NotImplementedError('deprecated, to remove')

    def DownloadData(self, request, context):
        compress_file_name = None
        try:
            folder = cfg['data_root']
            if 'file_root_dir' in request.options and request.options['file_root_dir'] == 'result':
                folder = cfg['results_root_dir']
            norm_path = os.path.normpath(request.data_path)
            dir_part = os.path.dirname(norm_path)
            basename = os.path.basename(norm_path)
            path = norm_path if os.path.isabs(norm_path) else os.path.join(folder, dir_part, basename)
            log.info(f'download {request.data_path} from {folder}, which in {path}')

            if 'compress' in request.options:
                compress = request.options['compress'].strip().lower()
                if compress in ('.tar.gz', '.tgz', 'tgz', 'gztar', 'tarball', 'tar.gz'):
                    import tarfile
                    compress_file_name = path + '.tar.gz'
                    archive = tarfile.open(compress_file_name, 'w|gz')
                    archive.add(path, arcname=basename)
                    archive.close()
                    path = compress_file_name
                elif compress in ('.zip', 'zip'):
                    compress_file_name = path + '.zip'
                    if os.path.isdir(path):
                        import shutil
                        shutil.make_archive(path, 'zip', root_dir=os.path.join(folder, dir_part), base_dir=basename)
                    else:
                        from zipfile import ZipFile
                        with ZipFile(compress_file_name, 'w') as zipf:
                            zipf.write(path, arcname=basename)
                    path = compress_file_name
                else:
                    log.error(f'unsupported compress type: {compress}')
                    yield data_svc_pb2.DownloadReply(status=data_svc_pb2.TaskStatus.Failed)
                    return
            else:
                if os.path.isdir(path):
                    log.error(f'this is a directory, set compress type please')
                    yield data_svc_pb2.DownloadReply(status=data_svc_pb2.TaskStatus.Failed)
                    return
                else:
                    pass  # keep origin file

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
        finally:
            if compress_file_name:
                try:
                    os.remove(compress_file_name)
                except:
                    pass

    def SendSharesData(self, request, context):
        ans = data_svc_pb2.SendSharesDataReply(status=data_svc_pb2.TaskStatus.Cancelled)
        return ans

    def _send_dat(self, dat, to):
        log.info(f'async send data(size: {dat.size}) to {to}')
        with grpc.insecure_channel(to) as channel:
            stub = compute_svc_pb2_grpc.ComputeProviderStub(channel)
            req = compute_svc_pb2.UploadShardReq(task_id='0xC0DE01', data_id='0xDA7A1234')
            response = stub.UploadShard(req)
            log.info(response.ok)

    def ListData(self, request, context):
        log.info(context.peer())
        ans = data_svc_pb2.ListDataReply()
        folder = cfg['data_root']
        if not os.path.exists(folder):
            os.makedirs(folder)
        files = os.listdir(folder)
        for f in files:
            path = os.path.join(folder, f)
            sz = os.path.getsize(path)
            log.info(f'{f}: {sz}')
            row = ans.data.add()
            row.data_name = f
            row.size = sz
        ans.status = ERROR_CODE["OK"]
        ans.msg = 'list data success.'
        return ans

    def HandleTaskReadyGo(self, request, context):
        log.info(f'{context.peer()} submit a task {request.task_id}, thread id: {threading.get_ident()}')
        status, msg = self.task_manager.start(request)
        return common_pb2.TaskReadyGoReply(status=status, msg=msg)

    def HandleCancelTask(self, request, context):
        task_name = f'{request.task_id[:15]}-{request.party_id}'
        log.info(f'{context.peer()} want to cancel task {task_name}')
        status, msg = self.task_manager.cancel_task(request.task_id, request.party_id)
        log.info(f'cancel task {status}, {msg}')
        return common_pb2.TaskCancelReply(status=status, msg=msg)
