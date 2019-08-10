"""
The racetrack problem:
You drive a race car and you wish to finish the track as fast as possible.
The velocity of the car is discrete and contains two parts: horizontal and vertical velocities.
- The actions are increments to the velocity components of +1/-1/0 (9 total actions).
- Both velocity components are nonnegative and restricted to be less than 5.
- Both velocity components cannot be zero except for the start line.
- Each episode begins in random grid position with velocity of (0,0) and ends when the car crosses the finish line.
- If the car hits a track boundary it is moved back to a random position on the start line and both
  velocities are reduced to zero.
- With probability 0.1 at each time step the velocity increments are both zero independently of the intended
  increments.

We will compute the optimal policy in the two racetracks given in the exercise in the book.
"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from random import choice, uniform
from operator import add
from tracks import FIRST_TRACK, SECOND_TRACK, FIRST_TRACK_START, FIRST_TRACK_END, SECOND_TRACK_START, SECOND_TRACK_END

FIRST_TRACK_DIMS = (30, 32)
SECOND_TRACK_DIMS = (32, 17)

FIRST_TRACK_START_LEN = 23
FIRST_TRACK_END_LEN = 9

# Probability of failure, meaning  the velocity increments are zero independent of the chosen action.
PROB_FAILURE = 0.1

EPSILON = 0.1

MAX_VELOCITY = 5
MIN_VELOCITY = 0

# The fine that we take for each step that we make.
STEP_FINE = -1

MAX_EPISODE_LENGTH = 1000


def get_action(policy, state, available_actions, explore=True):
    coin_flip = uniform(0, 1)
    if explore and coin_flip < EPSILON:
        return choice(available_actions)
    return policy[state]


class MonteCarloRacer(object):
    """
    This class implements on policy monte carlo control by using a basic epislon-soft policy which we run
    the General Policy Improvement scheme on.
    This also contains the general properties of the MDP for the racetrack problem in order to save some code.
    """

    def __init__(self, racetrack):
        self.track = racetrack
        self.velocities = (0, 0)
        self.states = [(i, j, k, h) for i in range(racetrack.data.shape[0]) for j in range(racetrack.data.shape[1])
                       for k in range(5) for h in range(5)]
        self.actions = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]

        # Initialize a random policy.
        self.policy = {state: choice(self.actions) for state in self.states}
        self.q_values = {(state, action): float('-inf') for state in self.states for action in self.actions}

        # To save time get all invalid actions for start state -inf.
        # for state in self.track.start_list:
        #     for action in self.get_illegal_actions(state):
        #         self.q_values[(state, action)] = float('-inf')


    def is_valid(self, state, action):
        """
        Checks whether the given action is valid based on the current velocities.
        :param action:
        :return:
        """
        action_result = tuple(map(add, (state[2], state[3]), action))
        return max(action_result) < MAX_VELOCITY and (action_result[0] > 0 or action_result[1] > 0) \
                and min(action_result) >= MIN_VELOCITY

    def choose_valid_action(self, state, explore=True):
        possible_actions = self.get_possible_actions(state)
        actions = possible_actions if len(possible_actions) > 0 else self.actions
        chosen_action = get_action(self.policy, state, actions , explore)
        while not self.is_valid(state, chosen_action) and explore:
            chosen_action = get_action(self.policy, state, actions, explore)
        return chosen_action

    def transition(self, state, action):
        """
        Perform the transition in the given state based on our current policy,
         return the new state of the agent following the action.
        :param state: Current state
        :param action Selected action.
        :return:
        """

        # There is a probability of failure of the action in our model.
        coin_flip = uniform(0, 1)
        if coin_flip >= PROB_FAILURE:
            self.velocities = tuple(map(add, self.velocities, action))

        return (state[0] - self.velocities[0], state[1] + self.velocities[1]) + self.velocities

    def select_greedy_action(self, state):
        """
        Find the action which maximizes our q-value.
        :param state:
        :return:
        """
        max_q = float('-inf')
        chosen_action = None
        for action in self.actions:
            if self.q_values[(state, action)] > max_q:
                max_q = self.q_values[(state, action)]
                chosen_action = action

        return chosen_action

    def get_possible_actions(self, state):
        valid_actions = []
        for action in self.actions:
            if self.is_valid(state, action):
                post_action = (state[0] - action[0] - state[2], state[1] + action[1] + state[3])
                if not self.is_out_of_bounds(post_action):
                    valid_actions.append(action)
        return valid_actions

    def on_policy_monte_carlo(self, iters=1000):
        returns = {(state, action): [] for state in self.states for action in self.actions}
        for _ in range(iters):
            start_state = choice(self.track.start_list) + (0, 0)
            start_action = choice(self.get_possible_actions(start_state))
            self.velocities = (0, 0)
            total_reward = 0.0

            # Generate the episode and go over it backwards saving average returns for state-action pairs.
            episode = self.generate_episode(start_state, start_action)

            if len(episode) > 0:
                for state, action in reversed(episode):
                    total_reward += STEP_FINE
                    returns[(state, action)].append(total_reward)

                    # Update the Q-value and the policy to be epsilon-soft with respect to this value.
                    curr_returns = returns[(state, action)]
                    self.q_values[(state, action)] = sum(curr_returns) / len(curr_returns)
                    self.policy[state] = self.select_greedy_action(state)

    def did_cross_finish_line(self, state):
        return state[1] >= self.track.end_list["column"] and self.track.end_list["start_row"] <= state[0] <= \
               self.track.end_list["end_row"]

    def is_out_of_bounds(self, state):
        """
        Check if a given state is out of the bounds of the racetrack.
        :param state:
        :return:
        """
        return state[0] < 0 or state[1] < 0 or state[0] >= self.track.data.shape[0] or \
               state[1] >= self.track.data.shape[1] or self.track.data[state[0], state[1]] == 0

    def generate_episode(self, start_state, start_action, explore=True):
        """
        Based on the given policy generate an episode and a list of the state and actions ordered.
        We will put a limit on episode length to prevent infinite episodes.
        :param start_state Initial state of the episode.
        :param start_action Initial action chosen in the episode.
        :param explore Whether we use exploration or not via epsilon soft policy.
        :return:
        """
        episode = []
        curr_state = start_state
        curr_action = start_action
        while not self.did_cross_finish_line((curr_state[0], curr_state[1])) and len(episode) < MAX_EPISODE_LENGTH:
            episode.append((curr_state, curr_action))
            curr_state = self.transition((curr_state[0], curr_state[1]), curr_action)

            # Check if we are out of bounds of the racetrack.
            if self.is_out_of_bounds((curr_state[0], curr_state[1])) or len(self.get_possible_actions(curr_state)) == 0:
                curr_state = choice(self.track.start_list) + (0, 0)
                self.velocities = (0, 0)

            curr_action = self.choose_valid_action(curr_state, explore)

        return episode

    def draw_path(self):
        start_state = choice(self.track.start_list) + (0, 0)
        episode = self.generate_episode(start_state, get_action(self.policy, start_state, self.actions, False), False)
        delta = 1 / len(episode)

        cnt = 0
        for state, _ in episode:
            cnt += 1
            self.track.data[state[0], state[1]] = 1 - cnt * delta

        self.track.visualize()


class RaceTrack(object):
    def __init__(self, data, start_list, end_list):
        """
        Initialize the track with the data matrix that describes the race track and lists of tuples
        which represent the possible starting and ending indices.
        :param data:
        :param start_list:
        :param end_list:
        """
        self.data = data
        self.start_list = start_list
        self.end_list = end_list

    def visualize(self):
        plt.figure(figsize=(14, 6))
        plt.imshow(self.data, cmap='gray', interpolation='None')
        plt.show()


if __name__ == "__main__":
    first_track = RaceTrack(SECOND_TRACK, SECOND_TRACK_START, SECOND_TRACK_END)
    racer = MonteCarloRacer(first_track)
    racer.on_policy_monte_carlo(iters=10000)
    racer.draw_path()

