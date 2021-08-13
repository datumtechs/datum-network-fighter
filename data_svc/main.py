import logging
import multiprocessing as mp
import sys
from concurrent import futures
from signal import signal, SIGTERM

import grpc
from grpc_reflection.v1alpha import reflection

from common.consts import GRPC_OPTIONS
from common.report_engine import report_event
from common.task_manager import TaskManager
from common.utils import load_cfg
from config import cfg
from protos import data_svc_pb2, data_svc_pb2_grpc, via_svc_pb2
from svc import DataProvider


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg['thread_pool_size']), options=GRPC_OPTIONS)
    svc_provider = DataProvider(TaskManager(cfg))
    data_svc_pb2_grpc.add_DataProviderServicer_to_server(svc_provider, server)
    SERVICE_NAMES = (
        data_svc_pb2.DESCRIPTOR.services_by_name['DataProvider'].full_name,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port('[::]:%s' % cfg['port'])
    server.start()
    if cfg['pass_via']:
        from common.via_client import expose_me
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

    event_stop = mp.Event()
    report_process = mp.Process(target=report_event, args=(cfg['schedule_svc'], event_stop))
    report_process.start()


    def handle_sigterm(*_):
        print("Received shutdown signal")
        all_rpcs_done_event = server.stop(5)
        all_rpcs_done_event.wait(5)
        event_stop.set()
        print("Shut down gracefully")


    signal(SIGTERM, handle_sigterm)

    server.wait_for_termination()
    report_process.join()
    print('over')
