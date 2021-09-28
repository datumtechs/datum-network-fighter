import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent import futures
import grpc

from common.consts import GRPC_OPTIONS
from common.utils import load_cfg
from config import cfg
from protos.lib.api import sys_rpc_api_pb2
from protos.lib.api import sys_rpc_api_pb2_grpc
from protos import compute_svc_pb2
from protos import compute_svc_pb2_grpc
from protos import data_svc_pb2
from protos import data_svc_pb2_grpc
from svc import YarnService
from google.protobuf import empty_pb2


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

def client(cfg):
    time.sleep(1)  # wait the data_svc/compute_svc set up.
    data_svc_address = '{}:{}'.format(cfg["bind_ip"], cfg["data_svc_port"])
    conn = grpc.insecure_channel(data_svc_address)
    client = data_svc_pb2_grpc.DataProviderStub(channel=conn)
    response = client.GetStatus(empty_pb2.Empty())
    str_res = '{' + str(response).replace('\n', ' ').replace('  ', ' ').replace('{', ':{') + '}'
    log.info(f"get data svc status: {str_res}")
    
    compute_svc_address = '{}:{}'.format(cfg["bind_ip"], cfg["compute_svc_port"])
    conn = grpc.insecure_channel(compute_svc_address)
    client = compute_svc_pb2_grpc.ComputeProviderStub(channel=conn)
    response = client.GetStatus(empty_pb2.Empty())
    str_res = '{' + str(response).replace('\n', ' ').replace('  ', ' ').replace('{', ':{') + '}'
    log.info(f"get compute svc status: {str_res}")
    

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
    client(cfg)
    server.wait_for_termination()
    log.info('svc over')


if __name__ == '__main__':
    main()
