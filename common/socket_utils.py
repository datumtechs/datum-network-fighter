import socket
import random
from hashlib import sha1
from contextlib import contextmanager


@contextmanager
def get_free_loopback_tcp_port():
    if socket.has_ipv6:
        tcp_socket = socket.socket(socket.AF_INET6)
    else:
        tcp_socket = socket.socket(socket.AF_INET)
    tcp_socket.bind(('', 0))
    try:
        yield tcp_socket.getsockname()[1]
    finally:
        tcp_socket.close()


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def find_free_port_in_range(port_range: list) -> int:
    assert len(port_range) == 2, 'length of port_range must be 2.'
    start_port = int(port_range[0])
    end_port = int(port_range[1]) + 1
    port_list = list(range(start_port, end_port))
    random.shuffle(port_list)

    family = socket.AF_INET6 if socket.has_ipv6 else socket.AF_INET
    for port in port_list:
        tcp_socket = socket.socket(family)
        try:
            tcp_socket.bind(('', port))
            return port
        except:
            pass
        finally:
            tcp_socket.close()
    else:
        raise Exception(f'no port free in {port_range}')


def get_a_fixed_port(task_id: str, party_id: str) -> int:
    start_port = 49152
    end_port = 65535
    interval = end_port - start_port

    x = f'{task_id}_{party_id}'
    h = sha1(x.encode('utf-8')).hexdigest()
    port = int(h, 16) % interval + start_port
    return port
