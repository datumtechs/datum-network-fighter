import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from threading import Thread
from time import time

from keras import backend as K

from xiangqi import Camp, Env
from model_helper import (
    load_best_model_weight, reload_best_model_weight_if_changed,
    save_as_best_model)
from nn import NNModel
from data_helper import get_game_data_filenames, upload_data

logger = getLogger(__name__)


def start(cfg, io_channel):
    return SelfPlayWorker(cfg, io_channel).start()


class SelfPlayWorker:
    def __init__(self, config, io_channel):
        self.config = config
        self.io_channel = io_channel
        self.current_model = self.load_model()
        self.m = Manager()
        self.pipes_bundle = self.m.list([self.current_model.get_pipes(self.config.play.search_threads) for _ in
                                         range(self.config.play.max_processes)])
        self.buffer = []

    def start(self):
        self.buffer.clear()

        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.play.max_processes) as executor:
            for game_idx in range(self.config.play.max_processes * 2):
                futures.append(executor.submit(self_play_buffer, self.config, self.pipes_bundle))
            game_idx = 0
            while True:
                game_idx += 1
                start_time = time()
                x = futures.popleft()
                env, data = x.result()
                time_cost = time() - start_time
                winner = env.winner.name if env.winner is not None else str(env.winner)
                print(f'game: {game_idx}, time: {time_cost:.3f}, n_steps: {env.n_steps}, '
                      f'winner: {winner}, data len: {len(data)}')
                self.buffer += data
                if (game_idx % self.config.playdata.nb_game_in_file) == 0:
                    self.flush_buffer()
                    reload_best_model_weight_if_changed(self.current_model, self.io_channel)
                futures.append(executor.submit(self_play_buffer, self.config, self.pipes_bundle))  # Keep it going

    def load_model(self):
        model = NNModel(self.config)
        logger.info(f"create new model? {self.config.opts.new}")
        if self.config.opts.new or not load_best_model_weight(model, self.io_channel):
            logger.info(f"create model from scratch")
            model.build()
            save_as_best_model(model, self.io_channel)
        model.session = K.get_session()
        model.graph = model.session.graph
        return model

    def flush_buffer(self):
        """
        Flush the play data buffer and write the data to the appropriate location
        """
        rc = self.config.resource
        data = json.dumps(self.buffer)
        logger.info(f'data len: {len(data)}, type: {type(data)}')
        thread = Thread(target=upload_data, args=(data, self.config, self.io_channel, 'upload_playdata'))
        thread.start()
        self.buffer = []

    def remove_play_data(self):
        """
        Delete the play data from disk
        """
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.playdata.max_file_num:
            return
        for i in range(len(files) - self.config.playdata.max_file_num):
            os.remove(files[i])


def self_play_buffer(config, pipes_bundle):
    """
    Play one game and add the play data to the buffer
    :param Config config: config for how to play
    :param list(Connection) cur: list of pipes to use to get a pipe to send observations to for getting
        predictions. One will be removed from this list during the game, then added back
    :return (ChessEnv,list((str,list(float)): a tuple containing the final ChessEnv state and then a list
        of data to be appended to the SelfPlayWorker.buffer
    """
    pipes_strand = pipes_bundle.pop()  # borrow
    from player_mcts import MCTSPlayer

    players = {Camp.RED: MCTSPlayer(Camp.RED, config, pipes_strand=pipes_strand),
               Camp.BLACK: MCTSPlayer(Camp.BLACK, config, pipes_strand=pipes_strand)}
    env = Env()
    for p in players.values():
        p.env = env

    ob = env.reset()
    cnt, cost = 0, 0
    while True:
        # env.render()
        player = players[ob['cur_player']]
        start_time = time()
        action = player.make_decision(**ob)
        time_cost = time() - start_time
        cnt += 1
        cost += time_cost
        print(f'step: {cnt}, time cost(sec.): {time_cost:.3f}, average: {cost / cnt:.3f}')
        ob, reward, done, info = env.step(action)
        if done:
            extra = '' if info is None else f', info: {info}'
            print(f'player {player.id.name}, reward: {reward}{extra}')
            break
    print(f'average cost(sec.): {cost / cnt:.3f}')
    player = players[env.cur_player]
    player.finish_game(reward)
    oppo = players[env.cur_player.opponent()]
    oppo.finish_game(-reward)

    data = []
    for i in range(len(player.moves)):
        data.append(player.moves[i])
        if i < len(oppo.moves):
            data.append(oppo.moves[i])

    pipes_bundle.append(pipes_strand)
    return env, data
