#!/usr/bin/env python3
import numpy as np

"""
Agent that learns to play Rock Paper Scissors using 
Counterfactual Regret Minimization
Paper: http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
"""
class RPSAgent():
	"""
	Global Variables:
	- num_action : int
		Number of actions available to the user.

	- opp_startegy: np.array (1 x NUM_ACTIONS)
		Opponent strategy where each index is the 
		probability of taking that action

	Attributes:
	- strategy: np.array (1 x NUM_ACTIONS)
		i-th index represents the probability of doing
		action_i, in the case of RPS, it being rock, 
		paper, scissors respectively.

	- strategy_sum: np.array (1 x NUM_ACTIONS)
		Cummulative sum of strategy every iteration.


	"""
	num_action = 3
	num_iterations = int(100)
	
	def __init__(self, name):
		self.name = name
		self.regret_sum = np.asarray([0.0] * RPSAgent.num_action)
		self.strategy = np.asarray([0.0] * RPSAgent.num_action)
		self.strategy_sum = np.asarray([0.0] * RPSAgent.num_action)
		self.opp_strategy = np.asarray([0.0] * RPSAgent.num_action)

	def set_opponent(self, opponent):
		self.opp_strategy = opponent.get_strategy()
		
	def get_strategy(self):
		self.strategy = np.maximum(self.regret_sum, 0.0)
		normalizing_sum = np.sum(self.strategy)
		if normalizing_sum > 0:
			self.strategy /= normalizing_sum
		else:
			self.strategy = np.asarray([1 / RPSAgent.num_action] * RPSAgent.num_action)
		self.strategy_sum += self.strategy
		return self.strategy

	@staticmethod
	def get_action(strategy):
		return np.random.choice(RPSAgent.num_action, p=strategy)

	def train(self):
		self.regret_sum = np.asarray([0.0] * RPSAgent.num_action)
		action_utility = np.asarray([0.0] * RPSAgent.num_action)
		for i in range(RPSAgent.num_iterations):
			self.get_strategy()
			action = self.get_action(self.strategy)
			opp_action = self.get_action(self.opp_strategy)

			action_utility[opp_action] = 0
			action_utility[0 if opp_action == RPSAgent.num_action - 1 else opp_action + 1] = 1
			action_utility[RPSAgent.num_action - 1 if opp_action == 0 else opp_action - 1] = -1
			self.regret_sum += action_utility - action_utility[action]
		# print(f"{self.name} Self:", self.strategy)
		# print(f"{self.name} Opp:", self.opp_strategy)

	def get_avg_strategy(self):
		normalizing_sum = np.sum(self.strategy_sum)
		if normalizing_sum > 0:
			return self.strategy_sum / normalizing_sum
		else:
			return np.asarray([1 / self.num_action] * self.num_action)


if __name__ == "__main__":

	RPSAgent1 = RPSAgent("Player 1")
	RPSAgent2 = RPSAgent("Player 2")
	RPSAgent2.set_opponent(RPSAgent1)
	RPSAgent1.set_opponent(RPSAgent2)
	for i in range(1000):
		RPSAgent1.train()
		RPSAgent2.train()
	print(RPSAgent1.get_avg_strategy())
	print(RPSAgent2.get_avg_strategy())
