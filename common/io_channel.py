import logging

import grpc

from protos import net_comm_svc_pb2
from protos import net_comm_svc_pb2_grpc

print('enter io_channel.py')
log = logging.getLogger(__name__)


class IOChannel:
    def __init__(self, task, peers):
        self.task = task
        self.peers_ch = {}  # party_id: grpc channel
        self._cache_channel(peers)

    def _get_channel(self, target):
        return self.peers_ch[target]

    def _cache_channel(self, peers):
        for party_id, addr in peers.items():
            log.info(f'{party_id}: {addr}')
            channel = grpc.insecure_channel(addr)
            self.peers_ch[party_id] = channel

    def get_party_id(self):
        return self.task.party_id

    def get_task_id(self):
        return self.task.id

    def send_data(self, party_id, data):
        ch = self._get_channel(party_id)
        stub = net_comm_svc_pb2_grpc.NetCommProviderStub(ch)
        task_id = self.task.id
        response, call = stub.Send.with_call(
            net_comm_svc_pb2.SendReq(task_id=task_id, party_id=party_id, content=data),
            metadata=(
                ('task_id', task_id),
            ))
        return response

    def recv_data(self, party_id, task_id, timeout):
        return self.task.get_data(party_id)
