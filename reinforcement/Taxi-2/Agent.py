import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.episode_i = 1
        self.epsilon = 1/self.episode_i
	
    def get_e_gready_policy(self, state):
        greedy_position = np.argmax(self.Q[state])
        policy = [self.epsilon/self.nA]*self.nA
        policy[greedy_position] = 1 - self.epsilon + self.epsilon/self.nA
        return policy
      
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy = self.get_e_gready_policy(state)
        return np.random.choice(np.arange(self.nA), p=policy)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        alpha = .1
        gamma = 1
        if done:
            self.episode_i += 1
            self.epsilon = 1/self.episode_i
            self.Q[state][action] = self.Q[state][action] + alpha * (reward - self.Q[state][action])
        else:
            next_policy = self.get_e_gready_policy(next_state)
            self.Q[state][action] = self.Q[state][action] + alpha * (reward + gamma*np.dot(self.Q[next_state], next_policy) - self.Q[state][action])
            