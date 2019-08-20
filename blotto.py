import tqdm
import numpy as np

S = 7 # Number of Soldiers
N = 3 # Number of Battlefields

def normalize(A):
	norm_sum = np.sum(A)
	if norm_sum > 0:
		return A / norm_sum
	else:
		return np.ones(A.shape)/len(A)

def get_all_strategies(S, N):
	if N == 1:
		return [[S]]
	strategies = []
	for i in range(S + 1):
		strategies += [[i] + l for l in get_all_strategies(S-i, N-1)]
	return strategies

all_strats = np.asarray(get_all_strategies(S,N))
num_actions = len(all_strats)

def print_all():
	for i in range(num_actions):
		print(i, all_strats[i])

def get_utility(id_1, id_2):
	win = np.sum(all_strats[id_1] > all_strats[id_2])
	loss = np.sum(all_strats[id_1] < all_strats[id_2])
	return  win - loss

def get_action(distribution):
	return np.random.choice(len(distribution), p=distribution)

def get_regret(self_id, opp_id):
	utility = np.asarray([get_utility(i, opp_id) for i in range(num_actions)])
	return utility - utility[self_id]

def get_distribution(regret_sum):
	distribution = np.maximum(regret_sum, 0.0)
	return normalize(distribution)

def top_action(strategy):
	return all_strats[np.argmax(strategy)]

def train(num_iter):
	dist_sum_1 = np.zeros(num_actions)
	dist_sum_2 = np.zeros(num_actions)
	for i in tqdm.tqdm(range(num_iter // 10)):
		regret_sum_1 = np.zeros(num_actions)
		regret_sum_2 = np.zeros(num_actions)
		for j in range(num_iter):
			dist_1 = get_distribution(regret_sum_1)
			dist_sum_1 += dist_1
			dist_2 = get_distribution(regret_sum_2)
			dist_sum_2 += dist_2
			id_1 = get_action(dist_1)
			id_2 = get_action(dist_2)
			regret_sum_1 += get_regret(id_1, id_2)
			regret_sum_2 += get_regret(id_2, id_1)
	return normalize(dist_sum_1), normalize(dist_sum_2)

strat_1 , strat_2 = train(100)
print(top_action(normalize(strat_1)), strat_1[strat_1.argsort()[::-1][:3]])
print(top_action(normalize(strat_2)), strat_2[strat_2.argsort()[::-1][:3]])
