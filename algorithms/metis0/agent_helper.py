import numpy as np


def flip_policy(pol, cfg):
    return np.asarray([pol[ind] for ind in cfg.unflipped_index])


def flip_ucci_labels(labels):
    def repl(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])
    return [repl(x) for x in labels]


def is_black_turn(fen):
    return fen.split(" ")[1] == 'b'


def testeval(fen, absolute=False) -> float:
    piece_vals = {'K': 1, 'A': 2, 'B': 2, 'N': 4, 'R': 9, 'C': 4.5, 'P': 1}
    ans = 0.0
    tot = 0
    for c in fen.split(' ')[0]:
        if not c.isalpha():
            continue

        if c.isupper():
            ans += piece_vals[c]
            tot += piece_vals[c]
        else:
            ans -= piece_vals[c.upper()]
            tot += piece_vals[c.upper()]
    v = ans/tot
    if not absolute and is_black_turn(fen):
        v = -v
    assert abs(v) < 1
    return np.tanh(v * 3)  # arbitrary
