import os
import json
import logging

from common.consts import COMMON_EVENT
from common.socket_utils import get_free_loopback_tcp_port
import channel_sdk.grpc


log = logging.getLogger(__name__)

def build_io_channel_cfg(task_id, self_party_id, peers, data_party, compute_party, 
                         result_party, cfg, self_internal_addr):
    pass_via = cfg['pass_via']
    certs = cfg['certs']
    channel_log_level = cfg.get('channel_log_level', 2)
    certs_base_path = certs.get('base_path', '')
    config_dict = {'TASK_ID': task_id,
                   'ROOT_CERT': os.path.join(certs_base_path, certs['root_cert']),
                   'LOG_LEVEL': channel_log_level}

    list_node_info = []
    via_dict = {}
    for i, node_info in enumerate(peers):
        party_id = node_info.party_id
        addr = '{}:{}'.format(node_info.ip, node_info.port)
        if not pass_via:
            # if not use via, then internal_addr = via_addr. so via_addr must be unique
            self_internal_addr = addr
        # via_name = 'VIA{}'.format(i + 1)
        # via_dict[via_name] = addr
        for k, v in via_dict.copy().items():
            if addr == v:
                via_name = k
                break
        else:
            via_name = 'VIA{}'.format(i + 1)
            via_dict[via_name] = addr
        # log.info(f"via_name: {via_name}")
        if self_party_id == party_id:
            internal_addr = self_internal_addr
            server_sign_key = os.path.join(certs_base_path, certs['server_sign_key'])
            server_sign_cert = os.path.join(certs_base_path, certs['server_sign_cert'])
            server_enc_key = os.path.join(certs_base_path, certs['server_enc_key'])
            server_enc_cert = os.path.join(certs_base_path, certs['server_enc_cert'])
        else:
            internal_addr = ''
            server_sign_key = ''
            server_sign_cert = ''
            server_enc_key = ''
            server_enc_cert = ''
        # Clients of all parties must have certificates.
        client_sign_key = os.path.join(certs_base_path, certs['client_sign_key'])
        client_sign_cert = os.path.join(certs_base_path, certs['client_sign_cert'])
        client_enc_key = os.path.join(certs_base_path, certs['client_enc_key'])
        client_enc_cert = os.path.join(certs_base_path, certs['client_enc_cert'])
        one_node_info = dict(
            NODE_ID=party_id,
            NAME=node_info.name,
            ADDRESS=internal_addr,
            VIA=via_name,
            SERVER_SIGN_KEY=server_sign_key,
            SERVER_SIGN_CERT=server_sign_cert,
            SERVER_ENC_KEY=server_enc_key,
            SERVER_ENC_CERT=server_enc_cert,
            CLIENT_SIGN_KEY=client_sign_key,
            CLIENT_SIGN_CERT=client_sign_cert,
            CLIENT_ENC_KEY=client_enc_key,
            CLIENT_ENC_CERT=client_enc_cert   
        )
        list_node_info.append(one_node_info)

    config_dict['NODE_INFO'] = list_node_info
    config_dict['VIA_INFO'] = via_dict
    config_dict['DATA_NODES'] = data_party
    party = {p: f'P{i}' for i, p in enumerate(compute_party)}
    config_dict['COMPUTATION_NODES'] = party
    config_dict['RESULT_NODES'] = result_party
    return config_dict


def get_channel_config(task_id, self_party_id, peers, data_party, compute_party, result_party, 
                    cfg, event_type):
    parent_proc_ip = cfg['bind_ip']
    with get_free_loopback_tcp_port() as port:
        log.info(f'got a free port: {port}')
    self_internal_addr = f'{parent_proc_ip}:{port}'
    log.info(f'get a free port: {self_internal_addr}')
    config_dict = build_io_channel_cfg(task_id, self_party_id, peers, data_party, compute_party, 
                                       result_party, cfg, self_internal_addr)
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
