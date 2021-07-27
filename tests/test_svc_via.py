import logging
import multiprocessing as mp
import os
import unittest

import grpc
from common.socket_utils import get_free_loopback_tcp_port, is_port_in_use
from google.protobuf import empty_pb2

from test_helper import *


class SvcViaTest(unittest.TestCase):
    def setUp(self):
        def proc_via_svc_fn(cfg, event_ok, event_stop):
            os.chdir('../via_svc')
            import via_svc.main as via_main
            via_main.cfg.update(cfg)
            via_main.serve()
            event_ok.set()
            event_stop.wait()

        def proc_data_svc_fn(cfg, event_via, event_ok, event_stop):
            os.chdir('../data_svc')
            import data_svc.main as data_main
            data_main.cfg.update(cfg)
            event_via.wait()
            data_main.serve()
            event_ok.set()
            event_stop.wait()

        def proc_compute_svc_fn(cfg, event_via, event_ok, event_stop):
            os.chdir('../compute_svc')
            import compute_svc.main as compute_main
            compute_main.cfg.update(cfg)
            event_via.wait()
            compute_main.serve()
            event_ok.set()
            event_stop.wait()

        self.event_stop = mp.Event()

        with get_free_loopback_tcp_port() as port:
            print(f'got a free port for via_svc: {port}')
        self.via_svc_port = port
        args = {'public_ip': 'localhost',
                'port': self.via_svc_port
                }
        self.event_via_ok = mp.Event()
        proc_via_svc = mp.Process(target=proc_via_svc_fn, args=(
            args, self.event_via_ok, self.event_stop))

        with get_free_loopback_tcp_port() as port:
            print(f'got a free port for data_svc: {port}')
        self.data_svc_port = port
        args = {'bind_ip': 'localhost',
                'port': self.data_svc_port,
                'pass_via': True,
                'via_svc':  f'localhost:{self.via_svc_port}'
                }
        self.event_data_svc_ok = mp.Event()
        proc_data_svc = mp.Process(target=proc_data_svc_fn, args=(
            args, self.event_via_ok, self.event_data_svc_ok, self.event_stop))

        with get_free_loopback_tcp_port() as port:
            print(f'got a free port for compute_svc: {port}')
        self.compute_svc_port = port
        args = {'bind_ip': 'localhost',
                'port': self.compute_svc_port,
                'pass_via': True,
                'via_svc':  f'localhost:{self.via_svc_port}'
                }
        self.event_compute_svc_ok = mp.Event()
        proc_compute_svc = mp.Process(target=proc_compute_svc_fn, args=(
            args, self.event_via_ok, self.event_compute_svc_ok, self.event_stop))

        self.procs = [proc_via_svc, proc_data_svc, proc_compute_svc]
        for proc in self.procs:
            proc.start()

    def tearDown(self):
        self.event_stop.set()
        for proc in self.procs:
            proc.join()

    def test_svc_thru_via(self):
        self.event_via_ok.wait()
        self.assertTrue(is_port_in_use(self.via_svc_port))

        from protos import data_svc_pb2_grpc
        self.event_data_svc_ok.wait()
        self.assertTrue(is_port_in_use(self.data_svc_port))
        with grpc.insecure_channel(f'localhost:{self.via_svc_port}') as channel:
            stub = data_svc_pb2_grpc.DataProviderStub(channel)
            response = stub.GetStatus(empty_pb2.Empty())
        self.assertEqual(response.node_type, 'data_node')

        from protos import compute_svc_pb2_grpc
        self.event_compute_svc_ok.wait()
        self.assertTrue(is_port_in_use(self.compute_svc_port))
        with grpc.insecure_channel(f'localhost:{self.via_svc_port}') as channel:
            stub = compute_svc_pb2_grpc.ComputeProviderStub(channel)
            response = stub.GetStatus(empty_pb2.Empty())
        self.assertGreater(float(response.mem), 0)


if __name__ == '__main__':
    logging.basicConfig()
    unittest.main(verbosity=2)
