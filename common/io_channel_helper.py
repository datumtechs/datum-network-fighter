import os
import json
import logging

from common.consts import COMMON_EVENT
from common.socket_utils import find_free_port_in_range
import channel_sdk.pyio

log = logging.getLogger(__name__)


def build_io_channel_cfg(task_id, self_party_id, peers, data_party, compute_party,
                         result_party, cfg, current_server_ip):
    pass_ice = cfg['pass_ice']
    ice_grid = cfg['ice_grid']
    grid_ip, grid_port = ice_grid.replace(' ', '').splie(':')
    certs = cfg['certs']
    channel_log_level = cfg.get('channel_log_level', 2)
    certs_base_path = certs.get('base_path', '')
    config_dict = {'TASK_ID': task_id, 'ROOT_CERT': os.path.join(certs_base_path, certs.get('root_cert', '')),
                   'LOG_LEVEL': channel_log_level, 'SEND_TIMEOUT': 10, 'CONNECT_TIMEOUT': 120}
    list_node_info = []
    ice_dict = {}
    grice2_dict = {}
    ice_grid_dict = {}
    for i, node_info in enumerate(peers):
        party_id = node_info.party_id
        addr = '{}:{}'.format(node_info.ip, node_info.port)
        for k, v in ice_dict.copy().items():
            if addr == v:
                ice_name = k
                break
        else:
            ice_name = 'ICE{}'.format(i + 1)
            ice_dict[ice_name] = addr

        ip, port = addr.split(':')
        grice2_dict[ice_name] = {
            "APPNAME": "ChannelGlacier2",
            "IP": ip,
            "PORT": port
        }

        address, public_ip = "", ""
        if self_party_id == party_id:
            ice_grid_dict[ice_name] = {
                "APPNAME": "ChannelIceGrid",
                "IP": grid_ip,
                "PORT": grid_port
            }
            public_ip = current_server_ip
            if not pass_ice:
                address = current_server_ip
        one_node_info = dict(
            NODE_ID=party_id,
            ADDRESS=address,
            PUBLIC_IP=public_ip,
            GRICER2=ice_name,
            ICEGRID=ice_name,
            CERT_DIR=os.path.join(certs_base_path, certs.get('cert_dir', '')),
            SERVER_CERT=os.path.join(certs_base_path, certs.get('server_cert', '')),
            CLIENT_CERT=os.path.join(certs_base_path, certs.get('client_cert', '')),
            PASSWORD=os.path.join(certs_base_path, certs.get('password', ''))
        )
        list_node_info.append(one_node_info)

    config_dict['NODE_INFO'] = list_node_info
    config_dict['DATA_NODES'] = data_party
    party = {p: f'P{i}' for i, p in enumerate(compute_party)}
    config_dict['COMPUTATION_NODES'] = party
    config_dict['RESULT_NODES'] = result_party
    config_dict['GRICER2_INFO'] = grice2_dict
    config_dict['ICE_GRID_INFO'] = ice_grid_dict
    return config_dict


def get_channel_config(task_id, self_party_id, peers, data_party, compute_party, result_party,
                       cfg, event_type):
    parent_proc_ip = cfg['bind_ip']
    task_port_range = cfg['task_port_range']
    port = find_free_port_in_range(task_port_range)
    self_internal_addr = f'{parent_proc_ip}:{port}'
    log.info(f'get a free port: {self_internal_addr}')
    config_dict = build_io_channel_cfg(task_id, self_party_id, peers, data_party, compute_party,
                                       result_party, cfg, parent_proc_ip)
    channel_config = json.dumps(config_dict)
    # log.info(f'self_party_id: {self_party_id}, channel_config: {channel_config}')
    log.info("get channel config finish.")

    return channel_config


if __name__ == '__main__':
    peers = [{'party_id': 'p5',
              'name': 'PartyA(P0)',
              'ip': '192.168.16.153',
              'port': '30005'},
             {'party_id': 'p6',
              'name': 'PartyB(P1)',
              'ip': '192.168.16.153',
              'port': '30005'},
             {'party_id': 'p7',
              'name': 'PartyC(P2)',
              'ip': '192.168.16.153',
              'port': '30006'}]

    create_channel('task-1', 'p5', peers,
                   [], ['p5', 'p6', 'p7'], [],
                   True, '192.168.9.32', {}, {})
