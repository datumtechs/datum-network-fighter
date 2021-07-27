
import logging
import os
import unittest

import grpc
from common.socket_utils import get_free_loopback_tcp_port, is_port_in_use
from google.protobuf import empty_pb2

from test_helper import *


class ComputeSvcTest(unittest.TestCase):
    def setUp(self):
        os.chdir('../compute_svc')
        from compute_svc.main import cfg, serve
        with get_free_loopback_tcp_port() as port:
            print(f'got a free port for compute_svc: {port}')
        cfg['bind_ip'] = 'localhost'
        cfg['port'] = port
        cfg['pass_via'] = False
        self._port = port
        self._server = serve()

    def tearDown(self):
        self._server.stop(0)

    def test_start_svc(self):
        from protos import compute_svc_pb2_grpc
        self.assertTrue(is_port_in_use(self._port))
        with grpc.insecure_channel(f'localhost:{self._port}') as channel:
            stub = compute_svc_pb2_grpc.ComputeProviderStub(channel)
            response = stub.GetStatus(empty_pb2.Empty())
        self.assertGreater(float(response.mem), 0)


class ComputeSvcRpcTest(unittest.TestCase):
    def setUp(self):
        os.chdir('../compute_svc')
        from compute_svc.main import cfg, serve
        with get_free_loopback_tcp_port() as port:
            print(f'got a free port: {port}')
        cfg['bind_ip'] = 'localhost'
        cfg['port'] = port
        cfg['pass_via'] = False
        self._server = serve()
        from protos import compute_svc_pb2_grpc
        self.channel = grpc.insecure_channel(f'localhost:{port}')
        self.comp_stub = compute_svc_pb2_grpc.ComputeProviderStub(self.channel)

    def tearDown(self):
        self.channel.close()
        self._server.stop(0)

    def test_comp_get_status(self):
        self.assertEqual(True, comp_get_status(self.comp_stub))

    def test_comp_task_details(self):
        self.assertEqual(True, comp_task_details(['1'], self.comp_stub))


if __name__ == '__main__':
    logging.basicConfig()
    unittest.main(verbosity=2)
