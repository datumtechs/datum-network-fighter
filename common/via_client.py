import logging

import grpc

from common.consts import GRPC_OPTIONS
from protos import via_svc_pb2
from protos import via_svc_pb2_grpc

log = logging.getLogger(__name__)


def expose_me(cfg, task_id, svc_type, party_id):
    if 'via_svc' not in cfg:
        return

    with grpc.insecure_channel(cfg['via_svc'], options=GRPC_OPTIONS) as channel:
        stub = via_svc_pb2_grpc.ViaProviderStub(channel)
        req = via_svc_pb2.ExposeReq(task_id=task_id,
                                    svc_type=svc_type,
                                    party_id=party_id,
                                    ip=cfg['bind_ip'],
                                    port=int(cfg['port']))
        log.info(req)
        ans = stub.Expose(req, wait_for_ready=True)
        log.info(ans)
