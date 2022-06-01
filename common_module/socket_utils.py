import socket
import random
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
    
    if socket.has_ipv6:
        family = socket.AF_INET6
    else:
        family = socket.AF_INET
    
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
