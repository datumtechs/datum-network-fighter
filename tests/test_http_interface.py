import json
import base64
import os
import requests

data_url = 'http://192.168.235.144:8080'
compute_url = 'http://192.168.235.144:8081'


# region data_http
def get_http_data():
    # /data/getStatus
    req = requests.get(f'{data_url}/data/getStatus')
    print(req.text)
    # /data/listData
    req = requests.get(f'{data_url}/data/listData')
    print(req.text)


def download_file(path, data_id):
    req = requests.post(f'{data_url}/data/downLoadData', data=json.dumps({"data_id": data_id}))
    if not os.path.isfile(path):
        with open(path, 'w'):
            pass
    with open(path, 'r+b') as f:
        for info in req.content.decode().split('\n'):
            if info:
                pq = json.loads(info)
                result = pq['result']
                if result.get('content', None):
                    f.write(base64.b64decode(result['content']))
                else:
                    print('download over')


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
    meta = {"meta": {"file_name": os.path.basename(path), "columns": ['12']}}
    yield bytes(json.dumps(meta), encoding='utf8')


def batch_iterable(path):
    for root, _, files in os.walk(path):
        for file in files:
            yield from iterable(os.path.join(root, file))


def upload_file(path, batch=False):
    print('start sending content')

    if not batch:
        req = requests.post(f'{data_url}/data/uploadData', data=iterable(path))
        print(req.text)
    else:
        req = requests.post(f'{data_url}/data/batchUpload', data=batch_iterable(path))
        print(req.text)


# download_file('./a.csv', 'Iris_20210531-074842.csv')
# get_http_data()
# upload_file('./result.csv')


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
    if type_ == 'handleTaskReadyGo':
        req = requests.post(f'{compute_url}/compute/handleTaskReadyGo', data=json.dumps(params))
    print(req.text)


task_info = {
    'task_id': '1234567',
    'contract_id': '7654321',
    'data_id': '7777777777',
    'party_id': 1,
    'env_id': '9999999999',
    'peers': [{'ip': '11.11.11.11', 'port': 1234, 'party': 0, 'name': 'Tom'},
              {'ip': '22.22.22.22', 'port': 4567, 'party': 1, 'name': 'Jerry'},
              {'ip': '33.33.33.33', 'port': 4567, 'party': 1, 'name': 'Peter'},
              ]
}


# endregion
def run_test(data_or_compute):
    if data_or_compute == 'data':
        # download_file('./a.csv', 'Iris_20210531-074842.csv')
        get_http_data()
        # upload_file('./result.csv')
    elif data_or_compute == 'compute':
        http_compute('getStatus')
    else:
        get_http_data()
        http_compute('getStatus')


run_test('all')
