import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent import futures
import grpc

from config import cfg
from common_module.consts import GRPC_OPTIONS
from common_module.utils import load_cfg
from consul_client.api import get_consul_client_obj
from consul_client.health import health_grpc_check
from pb.carrier.api import sys_rpc_api_pb2, sys_rpc_api_pb2_grpc
from pb.fighter.api.compute import compute_svc_pb2, compute_svc_pb2_grpc
from pb.fighter.api.data import data_svc_pb2, data_svc_pb2_grpc
from tests.schedule_svc.svc import YarnService
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
    health_grpc_check.add_service(server, 'jobNode')
    bind_port = cfg['port']
    server.add_insecure_port('[::]:%s' % bind_port)
    server.start()
    log.info(f'Schedule Service work on port {bind_port}.')
    return server

def client(cfg):
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
    parser.add_argument('--use_consul', type=int, default=1) # 1: use consul, 0: not use consul
    args = parser.parse_args()
    cfg.update(load_cfg(args.config))
    if args.bind_ip:
        cfg['bind_ip'] = args.bind_ip
    if args.port:
        cfg['port'] = args.port
    log.info(f"cfg: {cfg}")
    if args.use_consul:
        log.info('get consul client obj.')
        consul_client_obj = get_consul_client_obj(cfg)
        assert consul_client_obj, f'get consul client obj fail, cfg is:{cfg}'
        assert consul_client_obj.register(cfg), f'schedule svc register to consul fail, cfg is:{cfg}'

    server = serve()
    # wait the data_svc/compute_svc set up and connect success.
    connect_succ = False
    waiting_time_limit = 300
    waiting_time = 0
    start_time = time.time()
    while (not connect_succ) and (waiting_time < waiting_time_limit):
        try:
            client(cfg)
            connect_succ = True
            log.info("connect to data_svc/compute_svc success.")
        except grpc._channel._InactiveRpcError as e:
            waiting_time = time.time() - start_time
        except:
            raise
    server.wait_for_termination()
    log.info('svc over')


if __name__ == '__main__':
    main()
