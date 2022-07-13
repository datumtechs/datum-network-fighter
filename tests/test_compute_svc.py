
import logging
import unittest

import grpc
import config
from common.socket_utils import get_free_loopback_tcp_port, is_port_in_use
from google.protobuf import empty_pb2

from test_helper import *
from common.task_manager import TaskManager
from common.utils import load_cfg


class ComputeSvcTest(unittest.TestCase):
    def setUp(self):
        os.chdir('../compute_svc')
        from compute_svc.main import cfg, serve
        cfg.update(load_cfg("config.yaml"))
        with get_free_loopback_tcp_port() as port:
            print(f'got a free port for compute_svc: {port}')
        cfg['register_ip'] = 'localhost'
        cfg['port'] = port
        cfg['pass_via'] = False
        self._port = port
        task_manager = TaskManager(cfg)
        self._server = serve(task_manager)

    def tearDown(self):
        self._server.stop(0)

    def test_start_svc(self):
        from lib import compute_svc_pb2_grpc
        self.assertTrue(is_port_in_use(self._port))
        with grpc.insecure_channel(f'localhost:{self._port}') as channel:
            stub = compute_svc_pb2_grpc.ComputeProviderStub(channel)
            response = stub.GetStatus(empty_pb2.Empty())
        self.assertGreater(float(response.mem), 0)


class ComputeSvcRpcTest(unittest.TestCase):
    def setUp(self):
        os.chdir('../compute_svc')
        from compute_svc.main import cfg, serve
        cfg.update(load_cfg("config.yaml"))
        with get_free_loopback_tcp_port() as port:
            print(f'got a free port: {port}')
        cfg['register_ip'] = 'localhost'
        cfg['port'] = port
        cfg['pass_via'] = False
        task_manager = TaskManager(cfg)
        self._server = serve(task_manager)
        from lib import compute_svc_pb2_grpc
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
