from collections import defaultdict
from functools import partialmethod

import grpc

import compute_svc_pb2_grpc
import data_svc_pb2_grpc
import net_comm_svc_pb2_grpc
import schedule_svc_pb2_grpc
import via_svc_pb2
import via_svc_pb2_grpc
from config import cfg


def expose_me(cfg, task_id, svc_type):
    if 'via_svc' not in cfg:
        return

    with grpc.insecure_channel(cfg['via_svc']) as channel:
        stub = via_svc_pb2_grpc.ViaProviderStub(channel)
        req = via_svc_pb2.ExposeReq(task_id=task_id,
                                    svc_type=svc_type,
                                    ip=cfg['bind_ip'],
                                    port=int(cfg['port']))
        ans = stub.Expose(req)
        print(ans.ok, ans.ip, ans.port)


def get_call_meta(context):
    meta = context.invocation_metadata()
    interest = defaultdict(str)
    interest.update({k: v for k, v in meta if k in {'task_id', 'party'}})
    return interest


def create_hosted_svc_cls(base_cls):
    attrs = {}
    rpcs = [a for a in dir(base_cls) if not a.startswith('__') and callable(getattr(base_cls, a))]
    print('rpc:', rpcs)

    def __my_init__(self, via_svc, svc_type):
        self.via_svc = via_svc
        self.svc_type = svc_type

    def __rpc__(self, request, context, fn):
        d = get_call_meta(context)
        task_id = d['task_id']
        party_id = d['party_id']
        print(f'task_id: {task_id}, party_id: {party_id} fn: {fn}')
        stub = self.via_svc.get_real_svc(self.svc_type, task_id, party_id)
        return getattr(stub, fn)(request)

    attrs['__init__'] = __my_init__
    for m in rpcs:
        attrs[m] = partialmethod(__rpc__, fn=m)

    return type('MyProvider', (), attrs)


class ViaProvider(via_svc_pb2_grpc.ViaProviderServicer):
    def __init__(self, server):
        self.server = server
        self.holders = {}

    def Expose(self, request, context):
        print(context.peer())
        if request.svc_type == via_svc_pb2.DATA_SVC:
            HostedServicer = create_hosted_svc_cls(data_svc_pb2_grpc.DataProviderServicer)
            data_svc_pb2_grpc.add_DataProviderServicer_to_server(
                HostedServicer(self, request.svc_type), self.server)
            svc_addr = f'{request.ip}:{request.port}'
            channel = grpc.insecure_channel(svc_addr)
            call_id = self._get_call_id(request.svc_type, request.task_id, request.party_id)
            self.holders[call_id] = channel
        elif request.svc_type == via_svc_pb2.COMPUTE_SVC:
            HostedServicer = create_hosted_svc_cls(compute_svc_pb2_grpc.ComputeProviderServicer)
            compute_svc_pb2_grpc.add_ComputeProviderServicer_to_server(
                HostedServicer(self, request.svc_type), self.server)
            svc_addr = f'{request.ip}:{request.port}'
            channel = grpc.insecure_channel(svc_addr)
            call_id = self._get_call_id(request.svc_type, request.task_id, request.party_id)
            self.holders[call_id] = channel
        elif request.svc_type == via_svc_pb2.SCHEDULE_SVC:
            print('no impl')
        elif request.svc_type == via_svc_pb2.NET_COMM_SVC:
            print('TODO')
        else:
            raise ValueError(f'unknown svc type: {request.svc_type}')
        return via_svc_pb2.ExposeAns(ok=True, ip=cfg['public_ip'], port=cfg['port'])

    def Off(self, request, context):
        call_id = self._get_call_id(request.svc_type, request.task_id, request.party_id)
        channel = self.holders[call_id]
        channel.close()
        del self.holders[call_id]

    def get_real_svc(self, svc_type, task_id, party_id):
        svc_stub = {via_svc_pb2.DATA_SVC: data_svc_pb2_grpc.DataProviderStub,
                    via_svc_pb2.COMPUTE_SVC: compute_svc_pb2_grpc.ComputeProviderStub,
                    via_svc_pb2.SCHEDULE_SVC: schedule_svc_pb2_grpc.ScheduleProviderStub,
                    via_svc_pb2.NET_COMM_SVC: net_comm_svc_pb2_grpc.NetCommProviderStub
                    }
        call_id = self._get_call_id(svc_type, task_id, party_id)
        channel = self.holders[call_id]
        stub = svc_stub[svc_type](channel)
        return stub

    def _get_call_id(self, svc_type, task_id, party_id):
        return f'{svc_type}:{task_id}:{party_id}'
