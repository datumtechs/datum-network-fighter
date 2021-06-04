import logging
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection

from config import cfg
from protos import data_svc_pb2
from protos import data_svc_pb2_grpc
from protos import via_svc_pb2
from svc import DataProvider


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    data_svc_pb2_grpc.add_DataProviderServicer_to_server(DataProvider(), server)
    SERVICE_NAMES = (
        data_svc_pb2.DESCRIPTOR.services_by_name['DataProvider'].full_name,
    )
    print(reflection.SERVICE_NAME)
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port('[::]:%s' % cfg['port'])
    server.start()
    if cfg['pass_via']:
        from via_svc.svc import expose_me
        expose_me(cfg, 'task-1', via_svc_pb2.DATA_SVC)
    print('Data Service ready for action.')
    server.wait_for_termination()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config for start up")
    parser.add_argument('config', help='start info')

    args = parser.parse_args()
    logging.basicConfig()

    serve()
