import os
import grpc
import unittest
import pandas as pd
import configparser
import time
from google.protobuf import empty_pb2
from protos import compute_svc_pb2, compute_svc_pb2_grpc
from protos import data_svc_pb2, data_svc_pb2_grpc
from protos import common_pb2
from test_helper import *

config = configparser.ConfigParser()
config.read('../gateway/common/config.ini')
task_id = '123456'


class UnitTest(unittest.TestCase):
    addr = f"{config.get('DataSvc', 'rpc_host')}:{config.get('DataSvc', 'rpc_port')}"
    ch = grpc.insecure_channel(addr)
    data_stub = data_svc_pb2_grpc.DataProviderStub(ch)

    addr = f"{config.get('ComputeSvc', 'rpc_host')}:{config.get('ComputeSvc', 'rpc_port')}"
    ch = grpc.insecure_channel(addr)
    comp_stub = compute_svc_pb2_grpc.ComputeProviderStub(ch)

    def test_get_status(self):
        self.assertEqual(True, get_status(self.data_stub, task_id))

    def test_upload(self):
        self.assertEqual(True, upload(self.data_stub, './test_data/p1.csv'))

    def test_upload_dir(self):
        self.assertEqual(True, upload_dir(self.data_stub, './test_data/'))

    def test_download(self):
        now = time.strftime("%Y%m%d-%H%M%S")
        self.assertEqual(True, download(self.data_stub, f'./test_data/download_{now}.csv', 'p0_20210623-124735.csv'))

    def test_list_data(self):
        self.assertEqual(True, list_data(self.data_stub))

    def test_comp_get_status(self):
        self.assertEqual(True, comp_get_status(self.comp_stub))

    def test_comp_task_details(self):
        self.assertEqual(True, comp_task_details(['1'], self.comp_stub))

    def test_comp_run_task(self):
        args = {
            'contract_id': '7654321',
            'data_id': '7777777777',
            'party_id': 1,
            'env_id': '9999999999',
            'peers': [{'ip': '192.168.235.151', 'port': 4565, 'party': 0, 'name': 'Tom'},
                      {'ip': '192.168.235.151', 'port': 4567, 'party': 1, 'name': 'Jerry'},
                      {'ip': '192.168.235.151', 'port': 4568, 'party': 2, 'name': 'Peter'},
                      ]
        }
        self.assertEqual(True, comp_run_task(args, self.comp_stub, task_id))


unittest.main(verbosity=1)
