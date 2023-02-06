"""PettingZoo environment for Hnefatafl.

Uses AECEnv from PettingZoo and the Hnefatafl library for the game logic."""
from typing import Any, Dict, List, Optional, Tuple
import gymnasium

import numpy as np

from gymnasium import spaces

from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.env import AECEnv

import hnefatafl as hn
from hnefatafl import utils as hn_utils


def env(render_mode=None) -> "HnefataflEnv":
    env = HnefataflEnv(render_mode=render_mode)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    if render_mode is not None:
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class HnefataflEnv(AECEnv):
    """PettingZoo environment for Hnefatafl.
    Inspired by the PettingZoo environment for Chess, modified for Hnefatafl.
    """
    metadata = {
        'render_modes': ['human'],
        'name': 'hnefatafl_v1',
        'is_parallelizable': False,
    }

    def __init__(self, render_mode: str = 'human'):
        super().__init__()

        self.board = hn.KingEscapeAndCaptureEasierBoard()
        """The Hnefatafl board."""

        self.agents = ['player_1', 'player_2']
        """The agents in the environment."""
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)

        self.action_spaces = {
            agent: spaces.Discrete(11 * 11 * 11 * 11) for agent in self.agents
        }
        """Space of possible actions. Each action is a tuple of four integers
        (x1, y1, x2, y2), where (x1, y1) is the starting position of the piece
        and (x2, y2) is the ending position of the piece."""

        # Define the observation space, including the "observation" and the "action mask"
        self.observation_spaces = {
            agent: spaces.Dict({
                'observation': spaces.Box(low=0, high=1, shape=(11, 11, 4), dtype=np.int8),
                'action_mask': spaces.Box(low=0, high=1, shape=(11 * 11 * 11 * 11,), dtype=np.int8)
            })
            for agent in self.agents
        }
        """Space of possible observations. The observation is a 11x11x3 array, where the first dimension are the friendly pieces,
        the second dimension are the enemy pieces, and the third dimension is the king location.
        The action mask is a 11x11x11x11 array, where the first two dimensions are the starting
        position of the piece, the second two dimensions are the ending position of the piece,
        and the value is 1 if the move is valid and 0 if the move is invalid."""

        self.rewards = None
        """The rewards for the agents."""

        self.infos = {agent: {} for agent in self.agents}
        """The infos for the agents."""

        self.truncations = {agent: False for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}

        self.agent_selection = None
        """The agent that is currently selecting a move."""

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        """The render mode."""

    def observation_space(self, agent):
        """Returns the observation space of the agent, using self.observation_spaces."""
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        """Returns the action space of the agent."""
        return self.action_spaces[agent]
    
    def observe(self, agent):
        """Returns the observation of the agent."""
        observation = hn_utils.get_observation(
            self.board, self.possible_agents.index(agent)
        )
        """The observation of the agent. Represents the board as a 11x11x4 array, where the first dimension are the friendly pieces,
        the second dimension are the enemy pieces, the third dimension is the king location, and the fourth dimension is the player color 
        (all zeros if the agent is player 1 and all ones if the agent is player 2)."""
        # TODO: try frame-stacking approach to observation, as in the Chess environment
        #   this will allow the neural network to learn the dynamics of the game by looking at
        #   the previous few frames
        legal_moves = (
            self.board.legal_moves if agent == self.agent_selection else []
        )
        action_mask = np.zeros(14641, "int8")
        for i in legal_moves:
            action = hn_utils.move_to_action(i)
            action_mask[action] = 1

        return {"observation": observation, "action_mask": action_mask}
    
    def reset(self, seed=None, return_info=False, options=None):
        """Resets the environment."""
        self.has_reset = True
        self.agents = self.possible_agents[:]
        self.board.reset()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        if self.render_mode == 'human':
            self.render()

    def set_game_result(self, result_val: int):  # result_val is -1 if black wins, 1 if white wins, 0 if draw
        for i, name in enumerate(self.agents):  # agent 0 is black, agent 1 is white
            self.terminations[name] = True
            if i == 0:
                self.rewards[name] = result_val
            else:
                self.rewards[name] = -result_val


    def step(self, action):
        """Performs a given action."""
        if (self.terminations[self.agent_selection] or
                self.truncations[self.agent_selection]):
            self._was_dead_step(None)
            return
        
        # Convert the action to a move.
        chosen_move = hn_utils.action_to_move(action)
        # check if the move is legal, if not, set the reward to -1 and terminate the game
        if chosen_move not in self.board.legal_moves:
            self.set_game_result(-1)
            self._accumulate_rewards()
            self.agent_selection = self._agent_selector.next()
            if self.render_mode == 'human':
                self.render()
            return
        self.board.push(chosen_move)

        # Check if the game is over and set the rewards and terminations.
        if outcome := self.board.outcome():
            result = outcome.result()
            result_val = hn_utils.result_to_int(result)
            self.set_game_result(result_val)

        self._accumulate_rewards()

        self.agent_selection = self._agent_selector.next()
        if self.render_mode == 'human':
            self.render()

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "Render was called, but no render mode was specified. "
            )
        elif self.render_mode == 'human':
            print(self.board)
        else:
            raise ValueError(
                f"Render mode {self.render_mode} is not supported."
            )
        
    def close(self):
        pass
