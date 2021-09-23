import logging
import multiprocessing as mp
import os
import sys
from concurrent import futures
import grpc

from common.consts import GRPC_OPTIONS
from common.utils import load_cfg
from config import cfg
from protos.lib.api import sys_rpc_api_pb2
from protos.lib.api import sys_rpc_api_pb2_grpc
from svc import YarnService


logging.basicConfig(
    level=logging.INFO,
    format='##### %(asctime)s %(levelname)-5s PID=%(process)-5d %(processName)-15s %(filename)-10s line=%(lineno)-5d %(name)-10s %(funcName)-10s: %(message)s',
    stream=sys.stderr
)
log = logging.getLogger(__name__)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg['thread_pool_size']), options=GRPC_OPTIONS)
    svc_provider = YarnService()
    sys_rpc_api_pb2_grpc.add_YarnServiceServicer_to_server(svc_provider, server)
    bind_port = cfg['port']
    server.add_insecure_port('[::]:%s' % bind_port)
    server.start()
    log.info(f'Schedule Service work on port {bind_port}.')
    return server


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='config.yaml')
    parser.add_argument('--bind_ip', type=str)
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    cfg.update(load_cfg(args.config))
    if args.bind_ip:
        cfg['bind_ip'] = args.bind_ip
    if args.port:
        cfg['port'] = args.port

    server = serve()
    server.wait_for_termination()
    log.info('svc over')


if __name__ == '__main__':
    main()
