import math
import random
from collections import namedtuple, deque
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####### 여기서부터 코드를 작성하세요 #######
# Actor 신경망을 구현해주세요!
class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_observations, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

# Critic 신경망을 구현해주세요!
class Critic(nn.Module):
    def __init__(self, n_observations):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_observations, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

####### 여기까지 코드를 작성하세요 #######
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.values = []
        self.next_states = []
        self.rewards = []
        self.length = 0

    def push(self, state, action, value, next_state, reward):
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.length += 1

    def clear(self):
        self.states = []
        self.actions = []
        self.values = []
        self.next_states = []
        self.rewards = []
        self.length = 0


class A2C:
    def __init__(self, state_size, action_size, gamma, lr_actor, lr_critic, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        self.episode_rewards = []
        self.memory = Memory()

        # 플로팅 초기화
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.reward_line, = self.ax.plot([], [], label='Total Reward')
        self.ax.legend()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Total Reward')

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = self.actor(state)
            value = self.critic(state)
        action = torch.multinomial(action_probs, num_samples=1)
        return action.item(), value.item()

    def plot_rewards(self):
        self.reward_line.set_xdata(range(len(self.episode_rewards)))
        self.reward_line.set_ydata(self.episode_rewards)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    ####### 여기서부터 코드를 작성하세요 #######
    # Actor-Critic의 업데이트를 구현해주세요!
    def update(self):
        states = torch.FloatTensor(self.memory.states).to(device)
        actions = torch.LongTensor(self.memory.actions).to(device)
        rewards = torch.FloatTensor(self.memory.rewards).to(device)
        next_states = torch.FloatTensor([s for s in self.memory.next_states if s is not None]).to(device)
        
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        
        returns = []
        advantages = []
        R = next_values[-1] if len(next_values) > 0 else 0
        
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R
            advantage = R - values[step]
            returns.insert(0, R)
            advantages.insert(0, advantage)
        
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        actor_loss = -(log_probs * advantages).mean() - 0.01 * entropy
        value_loss = F.mse_loss(values, returns)
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizer_actor.step()
        
        self.optimizer_critic.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer_critic.step()
        
        self.memory.clear()
    ####### 여기까지 코드를 작성하세요 #######