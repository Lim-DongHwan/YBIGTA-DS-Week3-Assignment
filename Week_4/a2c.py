import argparse
import gymnasium as gym
import torch
from itertools import count
from tqdm import tqdm
from assets import A2C, device
import numpy as np

def main(args):
    env = gym.make('LunarLander-v2', render_mode='None')
    state_size = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_size = env.action_space.shape[0]
    else:
        raise ValueError("Unsupported action space type")

    agent = A2C(state_size, action_size, args.gamma, args.lr, args.lr, args.tau)

    for i_episode in tqdm(range(args.episodes)):
        state, _ = env.reset()
        episode_reward = 0
        for t in count():
            action, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward # 에피소드 단위로 보상을 추적
            done = terminated or truncated
            
            agent.memory.push(state, action, value, next_state if not done else None, reward)
            
            state = next_state

            if done:
                if terminated and episode_reward >= 200:  # 성공적인 착지로 간주
                    print(f"Episode {i_episode}: Successful landing! Total reward: {episode_reward}")
                else:
                    print(f"Episode {i_episode}: Crash or failure. Total reward: {episode_reward}")
                agent.episode_rewards.append(episode_reward)
                agent.plot_rewards()
                break
            
            if len(agent.memory.states) >= args.update_step: # 배치 단위로 모델 업데이트 
                agent.update()

    print('Complete')

    # Save the model
    agent.actor.to('cpu')
    agent.critic.to('cpu')
    print('Saving model...')
    torch.save(agent.actor.state_dict(), args.save_path + '/longforyou23_actor.pth')
    torch.save(agent.critic.state_dict(), args.save_path + '/longforyou23_critic.pth')
    agent.actor.to(device)
    agent.critic.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train the agent')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient for target network')
    parser.add_argument('--update_step', type=int, default=10, help='Number of steps between updates')
    parser.add_argument('--save_path', type=str, default='.', help='Path to save the trained model')
    args = parser.parse_args()
    main(args)
