import logging
import threading

from protos import compute_svc_pb2, compute_svc_pb2_grpc

log = logging.getLogger(__name__)


class ComputeProvider(compute_svc_pb2_grpc.ComputeProviderServicer):
    def __init__(self, task_manager):
        self.task_manager = task_manager

    def GetStatus(self, request, context):
        return self.task_manager.get_sys_stat()

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
            if req.HasField('meta'):
                task_id = req.meta.task_id
                data_id = req.meta.data_id
                task = self.task_manager.get_task(task_id)
                result = compute_svc_pb2.UploadShardReply(ok=True, msg='got a shard head')
                yield result
            else:
                task.put_data(req.data)
                yield compute_svc_pb2.UploadShardReply(ok=True, msg='got data piece')

    def HandleTaskReadyGo(self, request, context):
        log.info(f'{context.peer()} submit a task, thread id: {threading.get_ident()}')
        ok, msg = self.task_manager.start(request)
        return compute_svc_pb2.UploadShardReply(ok=ok, msg=msg)
