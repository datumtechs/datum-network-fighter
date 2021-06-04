import logging
from concurrent import futures

import grpc

import via_svc_pb2_grpc
from config import cfg
from svc import ViaProvider


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    via_svc_pb2_grpc.add_ViaProviderServicer_to_server(ViaProvider(server), server)

    server.add_insecure_port('[::]:%s' % cfg['port'])
    server.start()
    print('I stand ready.')
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()

    serve()
