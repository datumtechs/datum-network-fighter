"""
Encapsulates the worker which evaluates newly-trained models and picks the best one
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from multiprocessing import Manager
from time import sleep
from typing import Tuple

import numpy as np
from keras import backend as K

from xiangqi import Camp, Env
from agent_helper import flip_policy, testeval
from model_helper import (load_best_model_weight, save_as_best_model)
from nn import NNModel
from player_mcts import MCTSPlayer
from data_helper import get_next_generation_model_dirs

logger = getLogger(__name__)


def start(config, io_channel):
    return EvaluateWorker(config, io_channel).start()


class EvaluateWorker:
    """
    Worker which evaluates trained models and keeps track of the best one

    Attributes:
        :ivar Config config: config to use for evaluation
        :ivar PlayConfig config: PlayConfig to use to determine how to play, taken from config.eval.play_config
        :ivar ChessModel current_model: currently chosen best model
        :ivar Manager m: multiprocessing manager
        :ivar list(Connection) pipes_bundle: pipes on which the current best ChessModel is listening which will be used to
            make predictions while playing a game.
    """

    def __init__(self, config, io_channel):
        """
        :param config: Config to use to control how evaluation should work
        """
        self.config = config
        self.io_channel = io_channel
        self.play_config = config.eval.play_config
        self.current_model = self.load_current_model()
        self.m = Manager()
        self.pipes_bundle = self.m.list([self.current_model.get_pipes(self.play_config.search_threads) for _ in
                                         range(self.play_config.max_processes)])

    def start(self):
        """
        Start evaluation, endlessly loading the latest models from the directory which stores them and
        checking if they do better than the current model, saving the result in self.current_model
        """
        while True:
            ng_model, model_dir = self.load_next_generation_model()
            logger.debug(f"start evaluate model {model_dir}")
            ng_is_great = self.evaluate_model(ng_model)
            if ng_is_great:
                logger.debug(f"New Model become best model: {model_dir}")
                save_as_best_model(ng_model, self.io_channel)
                self.current_model = ng_model
            self.move_model(model_dir)

    def evaluate_model(self, ng_model):
        """
        Given a model, evaluates it by playing a bunch of games against the current model.

        :param ChessModel ng_model: model to evaluate
        :return: true iff this model is better than the current_model
        """
        ng_pipes = self.m.list(
            [ng_model.get_pipes(self.play_config.search_threads) for _ in range(self.play_config.max_processes)])

        futures = []
        with ProcessPoolExecutor(max_workers=self.play_config.max_processes) as executor:
            for game_idx in range(self.config.eval.game_num):
                fut = executor.submit(play_game, self.config, cur=self.pipes_bundle, ng=ng_pipes,
                                      cur_red=(game_idx % 2 == 0))
                futures.append(fut)

            results = []
            for fut in as_completed(futures):
                # ng_score := if ng_model win -> 1, lose -> 0, draw -> 0.5
                ng_score, env, cur_red = fut.result()
                results.append(ng_score)
                win_rate = sum(results) / len(results)
                game_idx = len(results)
                logger.debug(f"game {game_idx:3}: ng_score={ng_score:.1f} as {'black' if cur_red else 'red'} "
                             f"win_rate={win_rate * 100:5.1f}% ")

                if len(results) - sum(results) >= self.config.eval.game_num * (1 - self.config.eval.replace_rate):
                    logger.debug(f"lose count reach {results.count(0)} so give up challenge")
                    return False
                if sum(results) >= self.config.eval.game_num * self.config.eval.replace_rate:
                    logger.debug(f"win count reach {results.count(1)} so change best model")
                    return True

        win_rate = sum(results) / len(results)
        logger.debug(f"winning rate {win_rate * 100:.1f}%")
        return win_rate >= self.config.eval.replace_rate

    def move_model(self, model_dir):
        """
        Moves the newest model to the specified directory

        :param file model_dir: directory where model should be moved
        """
        rc = self.config.resource
        new_dir = os.path.join(rc.next_generation_model_dir, "copies", os.path.basename(model_dir))
        os.makedirs(new_dir, exist_ok=True)
        os.rename(model_dir, new_dir)

    def load_current_model(self):
        """
        Loads the best model from the standard directory.
        :return ChessModel: the model
        """
        model = NNModel(self.config)
        ok = load_best_model_weight(model, self.io_channel)
        if not ok:
            logger.info("No best model found, start from scratch")
            model.build()
        model.session = K.get_session()
        model.graph = model.session.graph
        return model

    def load_next_generation_model(self):
        """
        Loads the next generation model from the standard directory
        :return (ChessModel, file): the model and the directory that it was in
        """
        rc = self.config.resource
        while True:
            dirs = get_next_generation_model_dirs(self.config.resource)
            if dirs:
                break
            logger.info("There is no next generation model to evaluate")
            sleep(60)
        model_dir = dirs[-1] if self.config.eval.evaluate_latest_first else dirs[0]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        model = NNModel(self.config)
        model.load(config_path, weight_path, self.io_channel)
        model.session = K.get_session()
        model.graph = model.session.graph
        return model, model_dir


def play_game(config, cur, ng, cur_red: bool) -> Tuple[float, Env, bool]:
    """
    Plays a game against models cur and ng and reports the results.

    :param Config config: config for how to play the game
    :param ChessModel cur: should be the current model
    :param ChessModel ng: should be the next generation model
    :param bool cur_red: whether cur should play red or black
    :return (float, ChessEnv, bool): the score for the ng model
        (0 for loss, .5 for draw, 1 for win), the env after the game is finished, and a bool
        which is true iff cur played as red in that game.
    """
    cur_pipes = cur.pop()
    ng_pipes = ng.pop()

    cur_model_camp = Camp.RED if cur_red else Camp.BLACK
    ng_model_camp = cur_model_camp.opponent()

    players = {cur_model_camp: MCTSPlayer(cur_model_camp, config, pipes_strand=cur_pipes, play_config=config.eval.play_config),
               ng_model_camp: MCTSPlayer(ng_model_camp, config, pipes_strand=ng_pipes, play_config=config.eval.play_config)}
    env = Env()
    for p in players.values():
        p.env = env

    ob = env.reset()
    while True:
        # env.render()
        player = players[ob['cur_player']]
        action = player.make_decision(**ob)
        ob, reward, done, info = env.step(action)
        if done:
            print(f'player {player.id.name}, reward: {reward}')
            break

    if env.winner == ng_model_camp:
        ng_score = 1
    elif env.winner == cur_model_camp:
        ng_score = 0
    else:
        ng_score = 0.5

    cur.append(cur_pipes)
    ng.append(ng_pipes)

    return ng_score, env, cur_red
