from math import exp, factorial
import random
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Implementation of a solution to Jack's Car rental problem.
Part of Chapter 4 : Dynamic Programming of Sutton & Barto Intro to Reinforcement Learning book.
The problem:
- Jack manages two locations for a nationwide car rental company.
- Each day customers arrive at locations to rent our cars.
- If Jack has a car he rents it out and he is credited 10$ for it.
- If he is out of cars at that location, the business is lost.
- Cars become available for renting the day after they are returned.
- Jack can move the cars between two locations overnight at the cost of 2$.
- We assume cars are rented and returned as a poisson random variable.
- There can be no more than 20 cars at each location.
- A max of 5 cars can be moved from one location to another.

Changes:
- One of jack's employees lives near the second location, she is happy to shuttle one of the cars
to the second location for free.
- Jack has limited parking space in each location. If more than 10 cars are kept overnight at a location
then an additional cost of 4$ must be incurred to use a second parking lot.

Notes:
- All policies here are deterministic.
- Discount rate will be 0.9, we assume this is a continuing finite MDP.
Lior Moshe 2019.
"""

MAX_NUM_CARS = 20

MOVING_REWARD = -2
RENTAL_REWARD = 10

# Expected number of rental/returns (used to compute poisson distributions).
FIRST_RENTAL = 3
SECOND_RENTAL = 4

FIRST_RETURNS = 3
SECOND_RETURNS = 2

DISCOUNT_RATE = 0.9

MAX_CARS_MOVED = 5

EPSILON = 1e-2

# Constant relevant to the changes in the problem.
MAX_CARS_AT_LOCATION = 10

SECOND_PARKING_LOT = 4


class DistType(Enum):
    RENTAL = 1,
    RETURN = 2


def compute_poison(expectation, min_val, max_val):
    """
    Compute the poisson distribution for the values between min and max where the expectation is
    the expected value (lambda in the poisson distribution computation).
    :param expectation:
    :param min_val:
    :param max_val:
    :return:
    """
    distribution = {}
    for i in range(min_val, max_val):
        distribution[i] = exp(-expectation) * (expectation ** i) / factorial(i)

    return distribution


def compute_rental_return_pairs(returns_dist, rentals_dist):
    """
    Given pair of distributions compute the probability of pairs from this distribution happenning.
    We assume that these two distributions are independent.
    :param returns_dist: Poisson distribution for number of return requests.
    :param rentals_dist: Poisson distribution for number of rental requests.
    :return:
    """
    pairs = []
    for rentals, rental_prob in rentals_dist.items():
        for returns, return_prob in returns_dist.items():
            pairs.append(RentalReturnPair(rentals, returns, rental_prob * return_prob))
    return pairs


class RentalReturnPair(object):

    def __init__(self, num_rental, num_return, prob):
        self.rental = num_rental
        self.returns = num_return
        self.prob = prob
        self.reward = num_rental * RENTAL_REWARD
        self.total = num_return + num_rental


class Policy(object):
    """
    Define a policy that given specific state will return distribution among available actions.
    Each state will be the number of cars at each location, the action will denote how many cars do we move between
    each location and how many do we sell, total reward will be calculated.
    An action will be how many cars do we move from first location to the second (negative means we move
    from the second to the first).
    """

    def __init__(self, with_ex_changes=False):
        # Generate random policy for each state.
        self._policy = {}
        self._value_func = {}
        for i in range(MAX_NUM_CARS):
            for j in range(MAX_NUM_CARS):
                # Each state is represented by number of cars at each location.
                curr_state = (i, j)
                self._policy[curr_state] = random.randint(-MAX_CARS_MOVED, MAX_CARS_MOVED)
                self._value_func[curr_state] = 0

        # Compute probability distribution for rental and returns in each location.
        self.first_dist = {DistType.RENTAL: {}, DistType.RETURN: {}}
        self.second_dist = {DistType.RENTAL: {}, DistType.RETURN: {}}

        self.first_dist[DistType.RENTAL] = compute_poison(FIRST_RENTAL, 0, MAX_NUM_CARS)
        self.first_dist[DistType.RETURN] = compute_poison(FIRST_RETURNS, 0, MAX_NUM_CARS)
        self.second_dist[DistType.RENTAL] = compute_poison(SECOND_RENTAL, 0, MAX_NUM_CARS)
        self.second_dist[DistType.RETURN] = compute_poison(SECOND_RETURNS, 0, MAX_NUM_CARS)

        # Compute probabilities of pairs of number of rentals and returns requests for each one of jack's locations.
        self.first_rental_returns = compute_rental_return_pairs(self.first_dist[DistType.RETURN],
                                                                self.first_dist[DistType.RENTAL])
        self.second_rental_returns = compute_rental_return_pairs(self.second_dist[DistType.RETURN],
                                                                 self.second_dist[DistType.RENTAL])

        self.changes = with_ex_changes

    def compute_next_possible(self, state, action):
        """
        Compute possible states following performing action on a given state.
        Given an action that states how many cars were moved we also need to take into account how many returns and
        rental requests jack can accept in each location (these will be weighted by the poisson distribution).
        :param state:
        :param action:
        :return:
        """
        states_and_rewards = []
        next_state = [0, 0]

        # Action x states x cars moved from first location to second (if negative it's the other way around.
        next_state[0] = state[0] - action
        next_state[1] = state[1] + action

        # If we work with the problem's changes we pay less because the worker can move one car for free.
        if self.changes:
            action -= 1
        base_reward = action * MOVING_REWARD

        # Now the state variates based on the poisson distribution that was computed in init.
        for first_pair in self.first_rental_returns:
            # todo- not sure about computation here what if we take into account order of returns and rentals requests?
            if first_pair.rental <= next_state[0] and first_pair.returns < MAX_NUM_CARS - next_state[0]:
                for second_pair in self.second_rental_returns:
                    if second_pair.rental <= next_state[1] and second_pair.returns < MAX_NUM_CARS - next_state[1]:
                        current_next_state = (next_state[0] - first_pair.rental + first_pair.returns,
                                              next_state[1] - second_pair.rental + second_pair.returns)
                        reward_add_on = 10 if (self.changes and (current_next_state[0] > MAX_CARS_AT_LOCATION
                                                                 or current_next_state[1] > MAX_CARS_AT_LOCATION)) \
                            else 0

                        states_and_rewards.append({"state": (next_state[0] - first_pair.rental + first_pair.returns,
                                                             next_state[1] - second_pair.rental + second_pair.returns),
                                                   "reward": base_reward +
                                                             RENTAL_REWARD * (first_pair.rental + second_pair.rental)
                                                             + reward_add_on,
                                                   "prob": first_pair.prob * second_pair.prob})
        return states_and_rewards

    def get_possible_states(self):
        return self._policy.keys()

    def get_chosen_action(self, state):
        try:
            return self._policy[state]
        except KeyError:
            return 0

    def evaluate(self):
        """
        Perform policy evaluation, given a policy compute it's value function.
        :return: Value function based on our policy.
        """
        delta = 0
        chosen_delta = 1000
        while chosen_delta > EPSILON:
            for state in self._policy.keys():
                action = self._policy[state];
                val = self._value_func[state];
                current_val = self.evaluate_action(state, action)
                self._value_func[state] = current_val
                chosen_delta = max(delta, abs(val - current_val))

        print("Finished evaluation")

    def evaluate_action(self, state, action):
        """
        Evaluate a given choice of action in a given state, check all the next possibilities using
        the poisson distribution of returns and rental that we computed in :func:`Policy.__init__`.
        :param state:
        :param action:
        :return:
        """
        current_val = 0.0
        possibilities = self.compute_next_possible(state, action)
        for possibility in possibilities:
            current_val += possibility["prob"] * \
                           (possibility["reward"] + DISCOUNT_RATE * self._value_func[possibility["state"]])
        return current_val

    @staticmethod
    def get_all_posssible_actions(state):
        max_first_move = min(state[0], MAX_CARS_MOVED)
        max_second_move = min(state[1], MAX_CARS_MOVED)
        actions = []

        for i in range(max_first_move + 1):
            for j in range(max_second_move + 1):
                actions.append(i - j)
        return actions

    def improvement(self):
        policy_stable = True

        for state in self._policy.keys():
            old_action = self._policy[state]
            possible_actions = Policy.get_all_posssible_actions(state)

            max_value = 0
            max_action = old_action
            for action in possible_actions:
                current_action_value = self.evaluate_action(state, action)

                if current_action_value > max_value:
                    max_value = current_action_value
                    max_action = action
            self._policy[state] = max_action
            if max_action != old_action:
                policy_stable = False
        return policy_stable

    @staticmethod
    def policy_iteration():
        policy = Policy()

        stability = False
        while not stability:
            policy.evaluate()
            stability = policy.improvement()
        print("Optimal policy!")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        first_nums = []
        second_nums = []
        resulting_action = []

        for state in policy.get_possible_states():
            first_nums.append(state[0])
            second_nums.append(state[1])
            resulting_action.append(policy.get_chosen_action(state))

        ax.scatter(first_nums, second_nums, resulting_action, c='r', marker='o')
        ax.set_xlabel("Number of cars in first location")
        ax.set_ylabel("Number of cars in second location")
        ax.set_zlabel("Number of cars moved from first location to second")

        plt.show()


if __name__ == "__main__":
    Policy.policy_iteration()
