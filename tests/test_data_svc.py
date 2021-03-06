import logging
import os
import tempfile
import time
import unittest
from unittest.mock import patch,MagicMock

import grpc
from common.socket_utils import get_free_loopback_tcp_port, is_port_in_use
from google.protobuf import empty_pb2

from test_helper import *
from common.task_manager import TaskManager
from common.utils import load_cfg


class DataSvcTest(unittest.TestCase):
    def setUp(self):
        os.chdir('../data_svc')
        from data_svc.main import cfg, serve
        cfg.update(load_cfg("config.yaml"))
        with get_free_loopback_tcp_port() as port:
            print(f'got a free port: {port}')
        cfg['port'] = port
        cfg['pass_via'] = False
        self._cfg = cfg
        task_manager = TaskManager(cfg)
        self._server = serve(task_manager)

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


class DataSvcRpcTest(unittest.TestCase):
    def setUp(self):
        os.chdir('../data_svc')
        from data_svc.main import cfg, serve
        cfg.update(load_cfg("config.yaml"))
        with get_free_loopback_tcp_port() as port:
            print(f'got a free port: {port}')
        cfg['port'] = port
        cfg['pass_via'] = False
        task_manager = TaskManager(cfg)
        self._server = serve(task_manager)
        from protos import data_svc_pb2_grpc
        self.channel = grpc.insecure_channel(f'localhost:{port}')
        self.data_stub = data_svc_pb2_grpc.DataProviderStub(self.channel)
        self.data_root_dir = cfg['data_root']
        if not os.path.exists(self.data_root_dir):
            os.makedirs(self.data_root_dir, exist_ok=True)
        self.a_test_file_name = os.path.join(self.data_root_dir, 'abc.csv')
        self._prepare_data()
        self.uploaded_files = []

    def tearDown(self):
        self.channel.close()
        self._server.stop(0)
        self._clean_test_data()

    def _prepare_data(self):
        with open(self.a_test_file_name, 'w') as f:
            c = 'id,x,y\n2,2.71828,e\n'
            f.write(c)

    def _clean_test_data(self):
        os.remove(self.a_test_file_name)
        for f in self.uploaded_files:
            os.remove(f)

    def test_get_status(self):
        self.assertEqual(True, get_status(self.data_stub, 'T001'))

    @patch('svc.report_file_summary')
    def test_upload(self, mock_report_file_summary):
        mock_report_file_summary.return_value = MagicMock(status=0)
        with tempfile.NamedTemporaryFile('w+t', suffix='.csv') as f:
            c = 'id,x,y\n1,3.14,pi\n'
            f.write(c)
            f.seek(0)
            ok, file_name = upload(self.data_stub, f.name)
            self.uploaded_files.append(file_name)
            self.assertEqual(True, ok)
            mock_report_file_summary.assert_called_once()

    def test_upload_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contents = ['id,x,y\n1,3.14,pi\n', 'id,x,y\n1,2.71828,e\n']
            for c in contents:
                with tempfile.NamedTemporaryFile('w+t', dir=tmpdir, suffix='.csv', delete=False) as f:
                    f.write(c)
                    f.seek(0)
            ok, file_names = upload_dir(self.data_stub, tmpdir)
            self.uploaded_files.extend(file_names)
            self.assertEqual(True, ok)

    def test_download(self):
        now = time.strftime("%Y%m%d-%H%M%S")
        download_to = f'tmp_{now}.csv'
        self.assertEqual(True, download(
            self.data_stub, download_to, self.a_test_file_name))
        self.uploaded_files.append(download_to)

    def test_list_data(self):
        self.assertEqual(True, list_data(self.data_stub))


if __name__ == '__main__':
    logging.basicConfig()
    unittest.main(verbosity=2)
