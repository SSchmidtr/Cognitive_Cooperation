import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Environments.simplegridv2 import SimpleEnv
import numpy as np
import matplotlib.pyplot as plt

class PolicyGradientAgent:
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.state_size = env.observation_space['image'].shape
        self.action_size = env.action_space.n
        self.policy = {}

    def get_action(self, state):
        state_key = tuple(state.flatten())
        if state_key not in self.policy:
            self.policy[state_key] = np.zeros(self.action_size)
        probs = self.softmax(self.policy[state_key])
        return np.random.choice(self.action_size, p=probs)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Resta el máximo para estabilidad numérica
        return exp_x / exp_x.sum()

    def update_policy(self, episode):
        discounted_rewards = []
        for t in range(len(episode)):
            Gt = sum([r * (self.gamma ** i) for i, (_, _, r) in enumerate(episode[t:])])
            discounted_rewards.append(Gt)
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)

        for t, (state, action, _) in enumerate(episode):
            state_key = tuple(state.flatten())
            if state_key not in self.policy:
                self.policy[state_key] = np.zeros(self.action_size)
            
            # Calcular el gradiente
            probs = self.softmax(self.policy[state_key])
            grad = -probs
            grad[action] += 1
            
            # Actualizar la política
            self.policy[state_key] += self.learning_rate * discounted_rewards[t] * grad

    def train(self, num_episodes=10):
        reward_history = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_memory = []
            total_reward = 0
            while not done:
                action = self.get_action(state['image'])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_memory.append((state['image'], action, reward))
                state = next_state
                total_reward += reward
            self.update_policy(episode_memory)
            reward_history.append(total_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        print("Training completed.")
        return reward_history

def main():
    env = SimpleEnv(render_mode="human")
    agent = PolicyGradientAgent(env)
    reward_history = agent.train()
    
    plt.plot(reward_history)
    plt.title("Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    # Test the trained agent
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state['image'])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main()