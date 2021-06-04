import logging
import sys
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection

import net_comm_svc_pb2_grpc
from config import cfg
from net_comm_svc import NetCommProvider
from protos import compute_svc_pb2
from protos import compute_svc_pb2_grpc
from protos import net_comm_svc_pb2
from protos import via_svc_pb2
from svc import ComputeProvider
from task_manager import TaskManager


def serve(bind_port):
    task_manager = TaskManager()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    compute_svc_pb2_grpc.add_ComputeProviderServicer_to_server(ComputeProvider(task_manager), server)
    net_comm_svc_pb2_grpc.add_NetCommProviderServicer_to_server(NetCommProvider(task_manager), server)
    SERVICE_NAMES = (
        compute_svc_pb2.DESCRIPTOR.services_by_name['ComputeProvider'].full_name,
        net_comm_svc_pb2.DESCRIPTOR.services_by_name['NetCommProvider'].full_name,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port('[::]:%s' % bind_port)

    server.start()
    if cfg['pass_via']:
        from via_svc.svc import expose_me
        expose_me(cfg, 'task-1', via_svc_pb2.COMPUTE_SVC)
    print(f'Compute Service work on port {bind_port}.')
    server.wait_for_termination()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bind_port', type=int, default=cfg['port'])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='##### PID %(process)-8d %(processName)-15s %(filename)10s line %(lineno)-5d %(name)10s %(funcName)-10s: %(message)s',
        stream=sys.stderr,
    )

    serve(args.bind_port)
