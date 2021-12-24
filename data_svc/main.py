import logging
import multiprocessing as mp
import os
import sys
import threading
from concurrent import futures
from signal import signal, SIGTERM

import grpc
from grpc_reflection.v1alpha import reflection

cur_dir = os.path.abspath(os.path.dirname(__file__))
par_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
sys.path.insert(0, par_dir)
sys.path.insert(0, os.path.join(par_dir, 'protos'))
sys.path.insert(0, cur_dir)
from common.consts import GRPC_OPTIONS
from common.report_engine import report_task_event
from common.task_manager import TaskManager
from common.utils import load_cfg
from config import cfg
from protos import data_svc_pb2, data_svc_pb2_grpc
from svc import DataProvider
from consul_client.api import get_consul_client_obj

logging.basicConfig(
    level=logging.INFO,
    format='##### %(asctime)s %(levelname)-5s PID=%(process)-5d %(processName)-15s %(filename)-10s line=%(lineno)-5d %(name)-10s %(funcName)-10s: %(message)s',
    stream=sys.stderr
)
log = logging.getLogger(__name__)

def serve(task_manager):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg['thread_pool_size']), options=GRPC_OPTIONS)
    svc_provider = DataProvider(task_manager)
    data_svc_pb2_grpc.add_DataProviderServicer_to_server(svc_provider, server)
    SERVICE_NAMES = (
        data_svc_pb2.DESCRIPTOR.services_by_name['DataProvider'].full_name,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port('[::]:%s' % cfg['port'])
    server.start()
    log.info('Data Service ready for action.')
    log.info(f"python version: {sys.version}")
    return server


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='config.yaml')
    parser.add_argument('--bind_ip', type=str)
    parser.add_argument('--port', type=int)
    parser.add_argument('--schedule_svc', type=str)
    args = parser.parse_args()
    cfg.update(load_cfg(args.config))
    if args.bind_ip:
        cfg['bind_ip'] = args.bind_ip
    if args.port:
        cfg['port'] = args.port
    if args.schedule_svc:
        cfg['schedule_svc'] = args.schedule_svc

    consul_client_obj = get_consul_client_obj(cfg)
    if not consul_client_obj:
        log.info(f'get consul client obj fail,cfg is:{cfg}')
        return

    if not consul_client_obj.register(cfg):
        log.info(f'compute svc register to consul fail,cfg is:{cfg}')
        return
    event_stop = mp.Event()
    task_manager = TaskManager(cfg)
    task_manager.consul_client = consul_client_obj

    event_stop = mp.Event()
    task_manager = TaskManager(cfg)

    def task_clean(task_manager, event_stop):
        while not event_stop.wait(60):
            task_manager.clean()

    t_task_clean = threading.Thread(target=task_clean, args=(task_manager, event_stop))
    t_task_clean.start()

    server = serve(task_manager)

    report_process = mp.Process(target=report_task_event, args=(cfg['schedule_svc'], event_stop), name='report_process')
    report_process.start()

    def handle_sigterm(*_):
        log.info("Received shutdown signal")
        all_rpcs_done_event = server.stop(5)
        all_rpcs_done_event.wait(5)
        event_stop.set()
        consul_client_obj.stop()
        log.info("Shut down gracefully")

    signal(SIGTERM, handle_sigterm)

    t_task_clean.join()
    server.wait_for_termination()
    report_process.join()
    log.info('svc over')


if __name__ == '__main__':
    main()
