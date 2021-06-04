import threading

import net_comm_svc_pb2
import net_comm_svc_pb2_grpc


class NetCommProvider(net_comm_svc_pb2_grpc.NetCommProviderServicer):
    def __init__(self, task_manager):
        self.task_manager = task_manager

    def Send(self, request_it, context):
        for req in request_it:
            task = self.task_manager.get_task(req.task_id)
            ok = task.put_data(req.party_id, req.content)
            yield net_comm_svc_pb2.SendReply(ok=ok)
