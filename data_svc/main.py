import logging
import sys
from concurrent import futures

import grpc
from common.task_manager import TaskManager
from common.utils import load_cfg
from grpc_reflection.v1alpha import reflection
from protos import data_svc_pb2, data_svc_pb2_grpc, via_svc_pb2

from config import cfg
from svc import DataProvider
from common.report_engine import report_event
from multiprocessing import Process


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg['thread_pool_size']))
    data_svc_pb2_grpc.add_DataProviderServicer_to_server(DataProvider(TaskManager(cfg)), server)
    SERVICE_NAMES = (
        data_svc_pb2.DESCRIPTOR.services_by_name['DataProvider'].full_name,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port('[::]:%s' % cfg['port'])
    server.start()
    if cfg['pass_via']:
        from via_svc.svc import expose_me
        expose_me(cfg, '', via_svc_pb2.DATA_SVC, '')
    print('Data Service ready for action.')
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
