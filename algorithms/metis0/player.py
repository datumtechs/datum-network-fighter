from xiangqi import Camp, Force, Env
from xiangqi.board import parse_action, ACTION_ALIAS, ROW_INDICATOR_ALIAS, RowIndicator, to_iccs_action
from xiangqi.constants import Action
from xiangqi.piece import piece_2_char, APPEAR_RED, APPEAR_BLACK, ENDC


class Player:
    def __init__(self, id_):
        self._id_ = id_
        self._env = None

    @property
    def id(self):
        return self._id_

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

    def make_decision(self, **kwargs):
        raise NotImplementedError('leave the chance to successors')

    def finish_game(self, reward):
        pass


class Human(Player):
    def __init__(self, id_):
        super().__init__(id_)

    def make_decision(self, **kwargs):
        valid_actions = kwargs['valid_actions']
        fen = kwargs['board_state']
        env = Env.from_fen(fen)
        camp = env.cur_player
        board = env.board
        if len(valid_actions) == 0:
            return {'action': Action.RESIGN}
        oppo_sue_draw = kwargs['sue_draw']

        while True:
            appear = APPEAR_RED if self.env.cur_player == Camp.RED else APPEAR_BLACK
            you = appear + 'You' + ENDC
            hint = f'opponent sue for peace, do {you} agree? ' if oppo_sue_draw else f'what do {you} want? '
            directive = input(hint)

            try:
                action = self.flex_action_input(directive, camp, board,
                                                sue_draw=oppo_sue_draw, valid_actions=valid_actions)
                return action
            except Exception as e:
                print(f'{e}, try again')

    @staticmethod
    def flex_action_input(directive: str, camp, board, **kwargs):
        pinyin = {
            'ju': piece_2_char(Camp.RED, Force.JU),
            'ma': piece_2_char(Camp.RED, Force.MA),
            'xiang': piece_2_char(Camp.RED, Force.XIANG),
            'shi': piece_2_char(Camp.RED, Force.SHI),
            'shuai': piece_2_char(Camp.RED, Force.SHUAI),
            'jiang': piece_2_char(Camp.BLACK, Force.SHUAI),
            'pao': piece_2_char(Camp.RED, Force.PAO),
            'bing': piece_2_char(Camp.RED, Force.BING),
            'zu': piece_2_char(Camp.BLACK, Force.BING),
            'ping': ACTION_ALIAS[Action.TRAVERSE],
            '=': ACTION_ALIAS[Action.TRAVERSE],
            'jin': ACTION_ALIAS[Action.ADVANCE],
            '+': ACTION_ALIAS[Action.ADVANCE],
            'tui': ACTION_ALIAS[Action.RETREAT],
            '-': ACTION_ALIAS[Action.RETREAT],
            'he': ACTION_ALIAS[Action.SUE_DRAW],
            'shu': ACTION_ALIAS[Action.RESIGN],
            'qian': ROW_INDICATOR_ALIAS[RowIndicator.FRONT],
            'zhong': ROW_INDICATOR_ALIAS[RowIndicator.MID],
            'hou': ROW_INDICATOR_ALIAS[RowIndicator.REAR],
            'er': ROW_INDICATOR_ALIAS[RowIndicator.SECOND],
            'san': ROW_INDICATOR_ALIAS[RowIndicator.THIRD],
            'si': ROW_INDICATOR_ALIAS[RowIndicator.FORTH],
            'wu': ROW_INDICATOR_ALIAS[RowIndicator.FIFTH],
            'front': ROW_INDICATOR_ALIAS[RowIndicator.FRONT],
            'mid': ROW_INDICATOR_ALIAS[RowIndicator.MID],
            'rear': ROW_INDICATOR_ALIAS[RowIndicator.REAR],
            'second': ROW_INDICATOR_ALIAS[RowIndicator.SECOND],
            'third': ROW_INDICATOR_ALIAS[RowIndicator.THIRD],
            'forth': ROW_INDICATOR_ALIAS[RowIndicator.FORTH],
            'fifth': ROW_INDICATOR_ALIAS[RowIndicator.FIFTH],
        }
        directive = directive.strip().lower()
        if kwargs['sue_draw']:
            agree = directive == 'yes'
            return {'action': Action.SUE_DRAW, 'act_param': agree}
        for k, v in pinyin.items():
            directive = directive.replace(k, v)
        directive = directive.replace(' ', '')

        if directive == ACTION_ALIAS[Action.SUE_DRAW]:
            return {'action': Action.SUE_DRAW}
        if directive == ACTION_ALIAS[Action.RESIGN]:
            return {'action': Action.RESIGN}

        piece, dst = parse_action(directive, camp, board)
        ucci_action = to_iccs_action({'piece': piece, 'dst': dst})
        return ucci_action


class NoBrain(Player):
    def __init__(self, id_):
        super().__init__(id_)

    def make_decision(self, **kwargs):
        valid_actions = kwargs['valid_actions']
        if len(valid_actions) == 0:
            return {'action': Action.RESIGN}
        if kwargs['sue_draw']:
            return {'action': Action.SUE_DRAW, 'act_param': True}
        import random
        action = random.choice(valid_actions)
        return action


class Playbook(Player):
    def __init__(self, id_, moves, result):
        super().__init__(id_)
        self.moves = moves
        self.step = 0
        self.result = result

    def make_decision(self, **kwargs):
        valid_actions = kwargs['valid_actions']
        fen = kwargs['board_state']
        env = Env.from_fen(fen)
        camp = env.cur_player
        board = env.board
        if len(valid_actions) == 0:
            return {'action': Action.RESIGN}
        if kwargs['sue_draw']:
            return {'action': Action.SUE_DRAW, 'act_param': True}

        if self.step == len(self.moves):
            if self.result == '1/2-1/2':
                return {'action': Action.SUE_DRAW}
            else:
                return {'action': Action.RESIGN}

        move = self.moves[self.step]
        print(f'{self.id.name} {move}')
        self.step += 1
        piece, dst = parse_action(move, camp, board)
        ucci_action = to_iccs_action({'piece': piece, 'dst': dst})
        return ucci_action
