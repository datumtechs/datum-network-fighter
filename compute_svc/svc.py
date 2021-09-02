import logging
import threading

import psutil

from protos import common_pb2, compute_svc_pb2, compute_svc_pb2_grpc

log = logging.getLogger(__name__)


def get_sys_stat():
    stat = compute_svc_pb2.GetStatusReply()
    _, _, load15 = psutil.getloadavg()
    stat.cpu = str(load15 / psutil.cpu_count() * 100)
    vm = psutil.virtual_memory()
    stat.mem = str(vm.percent)
    net = psutil.net_io_counters()
    b = net.bytes_sent + net.bytes_recv
    stat.bandwidth = str(b)
    return stat


class ComputeProvider(compute_svc_pb2_grpc.ComputeProviderServicer):
    def __init__(self, task_manager):
        self.task_manager = task_manager

    def GetStatus(self, request, context):
        return get_sys_stat()

    def GetTaskDetails(self, request, context):
        ret = compute_svc_pb2.GetTaskDetailsReply()
        for task_id in request.task_ids:
            task = self.task_manager.get_task(task_id)
            if task is not None:
                detail = ret.add()
                detail.task_id = task_id
                detail.elapsed_time = task.get_elapsed_time()
        return ret

    def UploadShard(self, request_it, context):
        for req in request_it:
            yield compute_svc_pb2.UploadShardReply(ok=False, msg='deprecated')

    def HandleTaskReadyGo(self, request, context):
        log.info(f'{context.peer()} submit a task {request.task_id}, thread id: {threading.get_ident()}')
        ok, msg = self.task_manager.start(request)
        return common_pb2.TaskReadyGoReply(ok=ok, msg=msg)

    def HandleCancelTask(self, request, context):
        log.info(f'{context.peer()} want to cancel task {request.task_id}')
        ok, msg = self.task_manager.cancel_task(request.task_id)
        return common_pb2.TaskCancelReply(ok=ok, msg=msg)
