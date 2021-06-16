import logging
import os
import sys
import unittest

import grpc
from google.protobuf import empty_pb2


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


class DataSvcTest(unittest.TestCase):
    def setUp(self):
        print(11111, 'cwd:', os.getcwd())
        sys.path.append(os.path.abspath('../protos'))
        sys.path.append(os.path.abspath('../data_svc'))
        print('\n'.join(sys.path))
        os.chdir('../data_svc')
        print(22222, 'cwd:', os.getcwd())
        from data_svc.main import cfg, serve
        cfg['pass_via'] = False
        cfg['port'] = 12345
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


class DataSvcTestVia(unittest.TestCase):
    def setUp(self):
        path_cp = sys.path.copy()
        print('\n'.join(path_cp))

        def set_via():
            sys.path.append(os.path.abspath('../protos'))
            sys.path.append(os.path.abspath('../via_svc'))
            print(11111, 'cwd', os.getcwd())
            os.chdir('../via_svc')
            print(22222, 'cwd', os.getcwd())
            import via_svc.main as via_main
            via_main.cfg['port'] = 23333
            via_main.cfg['public_ip'] = 'localhost'
            port = via_main.cfg['port']
            server = via_main.serve()
            return port, server

        def set_data():
            os.chdir('../data_svc')
            print(33333, 'cwd', os.getcwd())
            last = sys.path.pop()
            print('last:', last)
            sys.path.append(os.path.abspath('../protos'))
            sys.path.append(os.path.abspath('../data_svc'))
            print('\n'.join(sys.path))
            import data_svc.main as data_main
            data_main.cfg['pass_via'] = True
            data_main.cfg['port'] = 12346
            port = data_main.cfg['port']
            server = data_main.serve()
            return port, server

        self.via_svc_port, self.via_server = set_via()
        self.data_svc_port, self.data_server = set_data()

    def tearDown(self):
        self.data_server.stop(0)
        self.via_server.stop(0)

    def test_start_svc_thru_via(self):
        from protos import data_svc_pb2_grpc
        self.assertTrue(is_port_in_use(self.via_svc_port))
        self.assertTrue(is_port_in_use(self.data_svc_port))

        port = self.data_svc_port
        with grpc.insecure_channel(f'localhost:{port}') as channel:
            stub = data_svc_pb2_grpc.DataProviderStub(channel)
            response = stub.GetStatus(empty_pb2.Empty())
        self.assertEqual(response.node_type, 'data_node')


if __name__ == '__main__':
    logging.basicConfig()
    unittest.main(verbosity=2)
