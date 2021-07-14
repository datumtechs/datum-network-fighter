import json
import base64
import os
import socket
import requests
import unittest
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('../gateway/common/config.ini')
data_url = f"http://{config.get('DataSvc', 'http_host')}:{config.get('DataSvc', 'http_port')}"
compute_url = f"http://{config.get('ComputeSvc', 'http_host')}:{config.get('ComputeSvc', 'http_port')}"


# region data_http
def get_http_data_get_status():
    # /data/getStatus
    req = requests.get(f'{data_url}/data/getStatus')
    print(req.text)
    return req.status_code


def get_http_data_list_data():
    # /data/listData
    req = requests.get(f'{data_url}/data/listData')
    print(req.text)
    return req.status_code


def download_file(path, file_name):
    req = requests.post(f'{data_url}/data/downLoadData', data=json.dumps({"data_id": file_name}))
    if not os.path.isfile(path):
        with open(path, 'w'):
            pass
    if req.status_code != 200:
        return req.status_code
    with open(path, 'r+b') as f:
        for info in req.content.decode().split('\n'):
            if info:
                pq = json.loads(info)
                result = pq['result']
                if result.get('content', None):
                    f.write(base64.b64decode(result['content']))
                else:
                    print('download over')
    return req.status_code


def iterable(path):
    with open(path, 'rb') as content_file:
        chunk_size = 1024
        chunk = content_file.read(chunk_size)
        while chunk:
            base64_str = base64.b64encode(chunk).decode()
            content_info = {"content": base64_str}
            result = bytes(json.dumps(content_info), encoding='utf8')
            yield result
            chunk = content_file.read(chunk_size)

    df = pd.read_csv(path)
    file_name = os.path.basename(path)
    _, file_type = os.path.splitext(path)
    cols, dtypes = [], []
    for c, t in df.dtypes.items():
        cols.append(c)
        dtypes.append(str(t))
    meta = {"meta": {"file_name": file_name, "columns": cols, 'file_type': file_type, 'col_dtypes': dtypes}}
    yield bytes(json.dumps(meta), encoding='utf8')


def batch_iterable(path):
    for root, _, files in os.walk(path):
        for file in files:
            yield from iterable(os.path.join(root, file))


def upload_file(path, batch=False):
    print('start sending content')

    if not batch:
        req = requests.post(f'{data_url}/data/uploadData', data=iterable(path))
    else:
        req = requests.post(f'{data_url}/data/batchUpload', data=batch_iterable(path), headers={'Connection': 'close'})
    print(req.text)
    return req.status_code


# endregion
# region compute_http
def http_compute(type_, params=None):
    req = None
    if type_ == 'getStatus':
        # /compute/getStatus
        req = requests.get(f'{compute_url}/compute/getStatus')
    if type_ == 'getTaskDetails':
        # /compute/getTaskDetails
        req = requests.post(f'{compute_url}/compute/getTaskDetails', data=json.dumps(params))
    print(req.text)
    return req.status_code


def http_compute_task_go(params):
    req = requests.post(f'{compute_url}/compute/handleTaskReadyGo', data=json.dumps(params))
    print(req.text)
    return req.status_code


# endregion

class UnitTest(unittest.TestCase):
    def test_get_http_data_get_status(self):
        self.assertEqual(200, get_http_data_get_status())

    def test_get_http_data_list_data(self):
        self.assertEqual(200, get_http_data_list_data())

    def test_upload_file(self):
        self.assertEqual(200, upload_file('./test_data/p1.csv'))
        self.assertEqual(200, upload_file('./test_data/', True))

    def test_download_file(self):
        self.assertEqual(200, download_file('./test_data/test_download.csv', 'p_20210614-091658.csv'))

    def test_http_compute(self):
        self.assertEqual(200, http_compute('getStatus'))
        task_ids = {'task_ids': ['1']}
        self.assertEqual(200, http_compute('getTaskDetails', task_ids))

    def test_http_compute_task_go(self):
        task_info = {
            'task_id': 'Iris_20210612-184359.csv',
            'contract_id': '7654321',
            'data_id': '7777777777',
            'party_id': 1,
            'env_id': '9999999999',
            'peers': [{'ip': '192.168.235.151', 'port': 4565, 'party_id': 0, 'name': 'Tom'},
                      {'ip': '192.168.235.151', 'port': 4567, 'party_id': 1, 'name': 'Jerry'},
                      {'ip': '192.168.235.151', 'port': 4568, 'party_id': 2, 'name': 'Peter'},
                      ]
        }
        self.assertEqual(200, http_compute_task_go(task_info))


try:
    requests.get(data_url)
    requests.get(compute_url)
except socket.error as e:
    print(e)
else:
    unittest.main(verbosity=1)
