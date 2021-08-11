import logging
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection

from common.utils import load_cfg
from config import cfg
from protos import via_svc_pb2, via_svc_pb2_grpc
from svc import ViaProvider


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg['thread_pool_size']))
    via_svc_pb2_grpc.add_ViaProviderServicer_to_server(ViaProvider(server), server)
    SERVICE_NAMES = (
        via_svc_pb2.DESCRIPTOR.services_by_name['ViaProvider'].full_name,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port('[::]:%s' % cfg['port'])
    server.start()
    print('I stand ready.')
    return server


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config for start up")
    parser.add_argument('--config', help='new config')
    parser.add_argument('--port', type=int, help='port listen at')

    args = parser.parse_args()
    if args.config:
        cfg.update(load_cfg(args.config))
    if args.port:
        cfg['port'] = args.port

    logging.basicConfig()

    server = serve()
    server.wait_for_termination()
