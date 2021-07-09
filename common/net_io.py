import json
import logging
import socket
from contextlib import contextmanager
from typing import Iterable

import io_channel
import latticex.rosetta as rtt
from protos import via_svc_pb2

log = logging.getLogger(__name__)
channel = None


@contextmanager
def get_free_loopback_tcp_port() -> Iterable[str]:
    if socket.has_ipv6:
        tcp_socket = socket.socket(socket.AF_INET6)
    else:
        tcp_socket = socket.socket(socket.AF_INET)
    tcp_socket.bind(('', 0))
    address_tuple = tcp_socket.getsockname()
    yield f"localhost:{address_tuple[1]}"
    tcp_socket.close()


def build_io_channel_cfg(task_id, self_party_id, peers, data_party, compute_party, result_party, pass_via, self_internal_addr):
    config_dict = {'TASK_ID': task_id}

    list_node_info = []
    via_dict = {}
    for i, node_info in enumerate(peers):
        party_id = node_info['party']
        addr = '{}:{}'.format(node_info['ip'], node_info['port'])
        if not pass_via:
            addr = self_internal_addr
        via_name = 'VIA{}'.format(i+1)
        via_dict[via_name] = addr
        internal_addr = self_internal_addr if self_party_id == party_id else ''
        one_node_info = dict(
            NODE_ID=party_id,
            NAME=node_info['name'],
            ADDRESS=internal_addr,
            VIA=via_name
        )
        list_node_info.append(one_node_info)

    config_dict['NODE_INFO'] = list_node_info
    config_dict['VIA_INFO'] = via_dict
    config_dict['DATA_NODES'] = data_party
    party = {p: f'P{i}' for i, p in enumerate(compute_party)}
    config_dict['COMPUTATION_NODES'] = party
    config_dict['RESULT_NODES'] = result_party
    return config_dict


def error_callback(a, b, c, d, e):
    print('nodeid:{}, id:{}, errno:{}, error_msg:{}, ext_data:{}'.format(a, b, c, d, e))


def get_current_via_address(config_dict, current_node_id):
    list_node_info = config_dict['NODE_INFO']
    address = ''
    via = ''
    for node_info in list_node_info:
        nodeid = node_info['NODE_ID']
        if nodeid == current_node_id:
            address = node_info['ADDRESS']
            via = node_info['VIA']
            break

    if '' == via:
        return '', ''

    via_info = config_dict['VIA_INFO']
    via_address = via_info[via]

    return address, via_address


def reg_to_via(task_id, config_dict, node_id):
    address, via_address = get_current_via_address(config_dict, node_id)
    print('========cur addr:{}, cur via addr:{}'.format(address, via_address))
    arr_ = address.split(':')
    ip = arr_[0]
    port = arr_[1]
    cfg = {'via_svc': via_address, 'bind_ip': ip, 'port': port}
    from via_svc.svc import expose_me
    expose_me(cfg, task_id, via_svc_pb2.NET_COMM_SVC, node_id)


def rtt_set_channel(task_id, self_party_id, peers, data_party, compute_party, result_party, pass_via, parent_proc_ip):
    with get_free_loopback_tcp_port() as self_internal_addr:
        pass
    port = self_internal_addr.split(':')[-1]
    self_internal_addr = f'{parent_proc_ip}:{port}'
    print('get a free port:', self_internal_addr)

    config_dict = build_io_channel_cfg(
        task_id, self_party_id, peers, data_party, compute_party, result_party, pass_via, self_internal_addr)

    rtt_config = json.dumps(config_dict)
    # with open('tmp.json', 'w') as f:
    #     json.dump(config_dict, f)

    node_id = self_party_id
    global channel
    channel = io_channel.create_channel(node_id, rtt_config, error_callback)

    if pass_via:
        reg_to_via(task_id, config_dict, node_id)

    rtt.set_channel(channel)
    print('set channel succeed==================')


if __name__ == '__main__':
    peers = [{'party': 'p0',
              'name': 'PartyA(P0)',
              'ip': '192.168.16.153',
              'port': '10000'},
             {'party': 'p1',
              'name': 'PartyB(P1)',
              'ip': '192.168.16.153',
              'port': '20000'},
             {'party': 'p2',
              'name': 'PartyC(P2)',
              'ip': '192.168.16.153',
              'port': '30000'}]

    rtt_set_channel('task-1', 'p0', peers,
                    [], ['p0', 'p1', 'p2'], [],
                    True, '192.168.16.151')
