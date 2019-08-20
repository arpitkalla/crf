#!/usr/bin/env python3

# Finding the equilibria by using Chance-Sampling CFR
# Section 3 in http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
#
# Sample Output: num_iter = 1e6
# Average game value: -0.04358025
# 1 [0.46856216 0.53143784]
# 3p [0.29420072 0.70579928]
# 1pb [0.65422852 0.34577148]
# 3b [0.25 0.75]
# 2 [0.44005098 0.55994902]
# 1p [0.45360707 0.54639293]
# 2pb [0.58304307 0.41695693]
# 1b [0.75 0.25]
# 3 [0.42710326 0.57289674]
# 3pb [0.4390469 0.5609531]
# 2p [0.42384494 0.57615506]
# 2b [0.50015003 0.49984997]


import tqdm
import numpy as np 
from collections import OrderedDict

# Constants
PASS, BET = 0, 1
NUM_ACTION = 2

# Global Variables
node_map = {}

class Node:
	"""
	Attributes:
		- I : string
			Info Set containing the cards and history
		- strat : np.array
			Distribution for the actions
		- strat_sum : np.array
			Sum of the distribution of actions
	"""
	def __init__(self, I):
		self.I = I
		self.strat = np.zeros(NUM_ACTION)
		self.strat_sum = np.zeros(NUM_ACTION)
		self.regret_sum = np.zeros(NUM_ACTION)

	@staticmethod
	def normalize(A):
		norm_sum = np.sum(A)
		if norm_sum > 0:
			return A / norm_sum
		else:
			return np.ones(A.shape)/len(A)

	def get_strat(self, prob):
		self.strat = Node.normalize(np.maximum(self.regret_sum, 0.0))
		self.strat_sum += prob * self.strat 
		return self.strat

	def get_avg_strat(self):
		return Node.normalize(self.strat_sum)

	def __str__(self):
		return str(self.get_avg_strat())


def cfr(cards, h, p0, p1):
	plays = len(h)
	player = plays % 2
	opp = 1 - player

	# Compute playoff
	if plays > 1:
		terminal_pass = h[-1] == "p"
		double_bet = h[-2:] == "bb"
		is_player_higher = cards[player] > cards[opp]
		if terminal_pass:
			if h == "pp":
				return 1 if is_player_higher else -1
			return 1
		elif double_bet:
			return 2 if is_player_higher else -2
	I = str(cards[player]) + h

	# Create/Get info set node
	node = node_map.get(I)
	if node is None:
		node = Node(I)
		node_map[I] = node

	# Get strategy and node utility
	strat = node.get_strat(p0 if player == 0 else p1)
	util = np.zeros(NUM_ACTION)
	for i in range(NUM_ACTION):
		next_h = (h + "p") if i == 0 else (h + "b")
		if player == 0:
			util[i] = -cfr(cards, next_h, p0 * strat[i], p1)
		else:
			util[i] = -cfr(cards, next_h, p0, p1 * strat[i])
	node_util = strat.dot(util)

	# Accumulate counterfactual regret
	regret = util - node_util
	node.regret_sum = (regret * p0) if player == 0 else (regret * p1)
	return node_util

def train(num_iter):
	cards = [1, 2, 3]
	util = 0
	for i in tqdm.tqdm(range(num_iter)):
		np.random.shuffle(cards)
		util += cfr(cards, "", 1, 1)
	print(f"Average game value: {util / num_iter}")


def print_all(node_map):
	for key in node_map:
		print(key, str(node_map[key]))

if __name__ == "__main__":
	num_iter = int(1e5)
	train(num_iter)
	print_all(node_map)



