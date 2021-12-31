import logging
import multiprocessing as mp
import sys
import grpc
import threading
from concurrent import futures
from signal import signal, SIGTERM, SIGKILL

from grpc_reflection.v1alpha import reflection
from config import cfg
from common.consts import GRPC_OPTIONS
from common.report_engine import report_task_event
from common.task_manager import TaskManager
from common.utils import load_cfg, get_schedule_svc
from lib import data_svc_pb2, data_svc_pb2_grpc
from data_svc.svc import DataProvider
from consul_client.api import get_consul_client_obj
from consul_client.health import health_grpc_check
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
    health_grpc_check.add_service(server,"dataNode")
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
    parser.add_argument('--use_consul', type=int, default=1) # 1: use consul, 0: not use consul
    args = parser.parse_args()
    cfg.update(load_cfg(args.config))
    cfg['schedule_svc'] = ''
    if args.bind_ip:
        cfg['bind_ip'] = args.bind_ip
    if args.port:
        cfg['port'] = args.port
    if args.use_consul:
        consul_client_obj = get_consul_client_obj(cfg)
        assert consul_client_obj, f'get consul client obj fail, cfg is:{cfg}'
        assert consul_client_obj.register(cfg), f'data svc register to consul fail, cfg is:{cfg}'

        pipe = mp.Pipe()
        get_schedule = threading.Thread(target=get_schedule_svc, args=(cfg, consul_client_obj, pipe[1]))
        get_schedule.daemon = True
        get_schedule.start()
    else:
        cfg['schedule_svc'] = args.schedule_svc

    task_manager = TaskManager(cfg)
    event_stop = mp.Event()
    def task_clean(task_manager, event_stop):
        while not event_stop.wait(60):
            task_manager.clean()

    t_task_clean = threading.Thread(target=task_clean, args=(task_manager, event_stop))
    t_task_clean.start()

    server = serve(task_manager)

    if args.use_consul:
        report_process = mp.Process(target=report_task_event, args=(cfg, event_stop, pipe[0]), name='report_process')
    else:
        report_process = mp.Process(target=report_task_event, args=(cfg, event_stop), name='report_process')
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
