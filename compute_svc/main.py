import logging
import sys
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection
from protos import compute_svc_pb2, compute_svc_pb2_grpc, via_svc_pb2

from config import cfg
from common.utils import load_cfg
from svc import ComputeProvider
from common.task_manager import TaskManager
from common.report_engine import report_event
from multiprocessing import Process

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg['thread_pool_size']))
    compute_svc_pb2_grpc.add_ComputeProviderServicer_to_server(ComputeProvider(TaskManager(cfg)), server)
    SERVICE_NAMES = (
        compute_svc_pb2.DESCRIPTOR.services_by_name['ComputeProvider'].full_name,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    bind_port = cfg['port']
    server.add_insecure_port('[::]:%s' % bind_port)
    server.start()
    if cfg['pass_via']:
        from via_svc.svc import expose_me
        expose_me(cfg, '', via_svc_pb2.COMPUTE_SVC, '')
    print(f'Compute Service work on port {bind_port}.')
    return server


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='new config')

    args = parser.parse_args()
    if args.config:
        cfg.update(load_cfg(args.config))

    logging.basicConfig(
        level=logging.INFO,
        format='##### PID %(process)-8d %(processName)-15s %(filename)10s line %(lineno)-5d %(name)10s %(funcName)-10s: %(message)s',
        stream=sys.stderr,
    )

    server = serve()
    report_process = Process(target=report_event, args=(cfg['schedule_svc'],))
    report_process.start()
    server.wait_for_termination()
