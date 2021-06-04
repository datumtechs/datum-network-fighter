import numpy as np

print('enter rosetta_helper.py')

g_protocol = 'SecureNN'
g_io_ch = None
party_id = None
g_vars = {}


def set_io(io_ch):
    print('set new io')
    global g_io_ch
    g_io_ch = io_ch
    global party_id
    party_id = g_io_ch.get_party_id()


def activate(protocol):
    global g_protocol
    g_protocol = protocol


def private_input(party, data):
    pass


def mpc_add(a, b):
    p = party_id
    if p == 'P0':
        g_io_ch.send_data('P1', a)
    elif p == 'P1':
        g_io_ch.send_data('P0', b)
    return a + b


def mpc_mul(a, b):
    p = party_id
    print(p)
    t = g_io_ch.get_task_id()
    if p == 'P0':
        print(f'{t}: send a_ to P1')
        g_io_ch.send_data('P1', a)

    elif p == 'P1':
        print(f'{t}: send b_ to P0')
    else:
        print(f'{t}: send r_ to P1')
        print(f'{t}: send r_ to P2')
    return a * b


def split_data(data, n_parts):
    assert n_parts > 1
    m = np.array(data)
    rng = np.random.default_rng()
    parts = []
    s = np.zeros_like(m)
    for i in range(n_parts - 1):
        p = rng.random(m.shape)
        s += p
        parts.append(p)
    parts.append(m - s)
    return parts


def sharing(x):
    parts = split_data(x, 3)
    g_io_ch.send_data('P0', parts[0])
    g_io_ch.send_data('P1', parts[1])
    g_io_ch.send_data('P2', parts[2])


def psi(x, y):
    '''private set intersection'''
    pass


def reveal():
    pass
