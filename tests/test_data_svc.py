import logging
import os
import sys
import unittest
from unittest.mock import patch
from unittest import mock

import grpc
from common.socket_utils import get_free_loopback_tcp_port, is_port_in_use
from google.protobuf import empty_pb2
from test_helper import *
import time


class DataSvcTest(unittest.TestCase):
    def setUp(self):
        sys.path.append(os.path.abspath('../protos'))
        sys.path.append(os.path.abspath('../data_svc'))
        os.chdir('../data_svc')
        from data_svc.main import cfg, serve
        with get_free_loopback_tcp_port() as port:
            print(f'got a free port: {port}')
        cfg['port'] = port
        cfg['pass_via'] = False
        self._cfg = cfg
        self._server = serve()

    def tearDown(self):
        self._server.stop(0)

    def test_start_svc(self):
        from protos import data_svc_pb2_grpc
        port = self._cfg['port']
        self.assertTrue(is_port_in_use(port))
        with grpc.insecure_channel(f'localhost:{port}') as channel:
            stub = data_svc_pb2_grpc.DataProviderStub(channel)
            response = stub.GetStatus(empty_pb2.Empty())
        self.assertEqual(response.node_type, 'data_node')


class DataSvcViaTest(unittest.TestCase):
    def setUp(self):
        path_cp = sys.path.copy()

        def set_via():
            sys.path.append(os.path.abspath('../protos'))
            sys.path.append(os.path.abspath('../via_svc'))
            os.chdir('../via_svc')
            import via_svc.main as via_main
            with get_free_loopback_tcp_port() as port:
                print(f'got a free port for via: {port}')
            via_main.cfg['port'] = port
            via_main.cfg['public_ip'] = 'localhost'
            server = via_main.serve()
            return port, server

        def set_data_svc(via_port):
            os.chdir('../data_svc')
            sys.path.pop()
            sys.path.append(os.path.abspath('../protos'))
            sys.path.append(os.path.abspath('../data_svc'))
            import data_svc.main as data_main
            with get_free_loopback_tcp_port() as port:
                print(f'got a free port for data_svc: {port}')
            data_main.cfg['port'] = port
            data_main.cfg['pass_via'] = True
            data_main.cfg['via_svc'] = f'localhost:{via_port}'
            data_main.cfg['bind_ip'] = 'localhost'
            server = data_main.serve()
            return port, server

        self.via_svc_port, self.via_server = set_via()
        self.data_svc_port, self.data_server = set_data_svc(self.via_svc_port)

    def tearDown(self):
        self.data_server.stop(0)
        self.via_server.stop(0)

    def test_start_svc_thru_via(self):
        from protos import data_svc_pb2_grpc
        self.assertTrue(is_port_in_use(self.via_svc_port))
        self.assertTrue(is_port_in_use(self.data_svc_port))

        with grpc.insecure_channel(f'localhost:{self.via_svc_port}') as channel:
            stub = data_svc_pb2_grpc.DataProviderStub(channel)
            response = stub.GetStatus(empty_pb2.Empty())
        self.assertEqual(response.node_type, 'data_node')


class DataSvcRpcTest(unittest.TestCase):
    def setUp(self):
        sys.path.append(os.path.abspath('../protos'))
        sys.path.append(os.path.abspath('../data_svc'))
        os.chdir('../data_svc')
        from data_svc.main import cfg, serve
        with get_free_loopback_tcp_port() as port:
            print(f'got a free port: {port}')
        cfg['port'] = port
        cfg['pass_via'] = False
        self._server = serve()
        from protos import data_svc_pb2_grpc
        self.channel = grpc.insecure_channel(f'localhost:{port}')
        self.data_stub = data_svc_pb2_grpc.DataProviderStub(self.channel)

    def tearDown(self):
        self.channel.close()
        self._server.stop(0)

    def test_get_status(self):
        self.assertEqual(True, get_status(self.data_stub, 'T001'))

    @patch('svc.report_file_summary')
    def test_upload(self, mock_report_file_summary):
        import tempfile
        with tempfile.NamedTemporaryFile('w+t', suffix='.csv') as f:
            c = 'id,x,y\n1,3.14,pi\n'
            f.write(c)
            f.seek(0)
            self.assertEqual(True, upload(self.data_stub, f.name))
            mock_report_file_summary.assert_called_once()

    # def test_upload_dir(self):
    #     self.assertEqual(True, upload_dir(self.data_stub, './test_data/'))

    # def test_download(self):
    #     now = time.strftime("%Y%m%d-%H%M%S")
    #     self.assertEqual(True, download(self.data_stub, f'./test_data/download_{now}.csv', 'p0_20210623-124735.csv'))

    # def test_list_data(self):
    #     self.assertEqual(True, list_data(self.data_stub))


class RpcTest(unittest.TestCase):
    def setUp(self):
        def set_data_svc():
            sys.path.append(os.path.abspath('../protos'))
            sys.path.append(os.path.abspath('../data_svc'))
            os.chdir('../data_svc')
            from data_svc.main import cfg, serve
            with get_free_loopback_tcp_port() as port:
                print(f'got a free port for data_svc: {port}')
            cfg['port'] = port
            cfg['pass_via'] = False
            server = serve()
            return port, server

        def set_compute_svc():
            sys.path.append(os.path.abspath('../protos'))
            sys.path.append(os.path.abspath('../compute_svc'))
            os.chdir('../compute_svc')
            from data_svc.main import cfg, serve
            with get_free_loopback_tcp_port() as port:
                print(f'got a free port for compute_svc: {port}')
            cfg['port'] = port
            cfg['pass_via'] = False
            server = serve()
            return port, server

    def tearDown(self):
        pass


if __name__ == '__main__':
    logging.basicConfig()
    unittest.main(verbosity=2)
