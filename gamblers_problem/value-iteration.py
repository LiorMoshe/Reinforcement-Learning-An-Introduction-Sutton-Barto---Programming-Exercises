import random
import matplotlib.pyplot as plt

"""
Implementation of a value iteration algorithm for the Gambler's problem.
Definition of the problem.
- A gambler has some capital and he wishes to put at stake some of his money over a result of a coin flip.
- The gambler puts x amount of money.
- If it tails the gambler loses his stake, otherwise he takes his stake and wins amount of money that is equal
to his stake.
- The game ends in a win if the gambler reaches a capital of 100$.
- The game ends in a loss if the gambler loses all of his money.
- The game can be described as a finite discounted MDP where:
- The state of the game is the capital of the gambler.
- Actions are the stakes that the gambler can put.
- Reward is zero throughout the game and 1 if the gambler wins the game.
"""

# The maximal state of the gambler will be when he has a capital worth 99$.
MAX_STATE = 100

# Probability that defines the coin flip, this is the probability of coming with heads.
COIN_PROB = 0.25

# The only reward we will receive will be 1 when the gambler has a capital of 100$.
WINNER_REWARD = 1

DISCOUNT_FACTOR = 1

THETA = 1e-10


class GamblersMDP(object):
    """
    This class represents the MDP for the gambler problem, this is a discounted finite MDP.
    """

    def __init__(self):
        self.states_values = {}
        self.policy = {}
        for i in range(1, MAX_STATE):
            self.states_values[i] = 0
            self.policy[i] = random.randint(0, i)

        # Initialize the two dummy states of termination when we have 0 and 100$.
        self.states_values[0] = 0
        self.states_values[MAX_STATE] = 1

    @staticmethod
    def get_next_possible_states_and_rewards(state, action):
        """
        Given an action in a given state of the gambler there are two outcomes, he loses money or gets some money
        based on the probability of the coin flip. The only case where there is any reward is when the gambler
        wins the game (holds a capital of 100$).
        :param state: Current state of the gambler.
        :param action: Action that the gambler chose (how much money did he bid).
        :return:  List of pairs of states and rewards and their probabilities of happening (p function's value)
        """
        next_states = list()
        next_states.append({"state": state - action, "reward": 0, "prob": 1 - COIN_PROB})
        next_states.append({"state": state + action, "reward": 1 if state + action == MAX_STATE else 0,
                           "prob": COIN_PROB})
        return next_states

    def find_max_action(self, state):
        """
        Find the action that gives us maximal value, equivalent of the main step in the bellman optimality equation.
        :param state:
        :return:
        """
        action_range = min(state, MAX_STATE - state)
        max_value = 0
        max_action = None
        for action in range(action_range + 1):
            next_states = GamblersMDP.get_next_possible_states_and_rewards(state, action)
            action_value = 0
            for state_reward in next_states:
                action_value += state_reward["prob"] * (state_reward["reward"] +
                                                        DISCOUNT_FACTOR * self.states_values[state_reward["state"]])
            if action_value > max_value:
                max_value = action_value
                max_action = action
        return max_value, max_action

    def value_iteration(self):
        """
        Implementation of the value iteration algorithm.
        :return:
        """
        delta = 1
        while delta > THETA:
            print("Iterating over values current delta: ",delta)
            delta = 0
            for state in self.policy.keys():
                current_value = self.states_values[state]
                max_action_value = self.find_max_action(state)[0]
                self.states_values[state] = max_action_value
                delta = max(delta, abs(current_value - max_action_value))

        # After we converged we can compute our policy.
        for state in self.policy.keys():
            self.policy[state] = self.find_max_action(state)[1]

    def plot_policy(self):
        # states = self.policy.keys()
        # actions = self.policy.values()
        plt.figure()
        plt.bar(list(self.policy.keys()), list(self.policy.values()))

        plt.xlabel("Capital (state of the gambler)")
        plt.ylabel("Stake")
        plt.show()


if __name__ == "__main__":
    mdp = GamblersMDP()
    mdp.value_iteration()
    mdp.plot_policy()






