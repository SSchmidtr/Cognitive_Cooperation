import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Environments.dualbrain import DualBrainSimpleEnv

class DualBrainPolicyGradientAgent:
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.state_size = (env.height, env.width, 3)
        self.action_size = 4  # Limitamos a 4 acciones (0-3)
        self.policy_brain1 = {}
        self.policy_brain2 = {}

    def get_action(self, state):
        state_key_brain1 = tuple(state['brain1'].flatten())
        state_key_brain2 = tuple(state['brain2'].flatten())

        if state_key_brain1 not in self.policy_brain1:
            self.policy_brain1[state_key_brain1] = np.random.rand(self.action_size)
        if state_key_brain2 not in self.policy_brain2:
            self.policy_brain2[state_key_brain2] = np.random.rand(self.action_size)

        probs_brain1 = self.softmax(self.policy_brain1[state_key_brain1])
        probs_brain2 = self.softmax(self.policy_brain2[state_key_brain2])

        action_brain1 = np.random.choice(self.action_size, p=probs_brain1)
        action_brain2 = np.random.choice(self.action_size, p=probs_brain2)

        combined_action = (action_brain1 + action_brain2) % self.action_size

        return combined_action

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def update_policy(self, episode):
        discounted_rewards_brain1 = self.calculate_discounted_rewards([r[2]['brain1_reward'] for r in episode])
        discounted_rewards_brain2 = self.calculate_discounted_rewards([r[2]['brain2_reward'] for r in episode])

        for t, (state, action, _) in enumerate(episode):
            self.update_brain_policy(state['brain1'], action, discounted_rewards_brain1[t], self.policy_brain1)
            self.update_brain_policy(state['brain2'], action, discounted_rewards_brain2[t], self.policy_brain2)

    def update_brain_policy(self, state, action, discounted_reward, policy):
        state_key = tuple(state.flatten())
        if state_key not in policy:
            policy[state_key] = np.random.rand(self.action_size)

        probs = self.softmax(policy[state_key])
        grad = np.zeros(self.action_size)
        grad[action] = 1 - probs[action]
        policy[state_key] += self.learning_rate * discounted_reward * grad

    def calculate_discounted_rewards(self, rewards):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = sum([r * (self.gamma ** i) for i, r in enumerate(rewards[t:])])
            discounted_rewards.append(Gt)
        discounted_rewards = np.array(discounted_rewards)
        return (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)

def train(env, agent, num_episodes=10):
    reward_history = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_memory = []
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_memory.append((state, action, info))
            state = next_state
            total_reward += reward

        agent.update_policy(episode_memory)
        reward_history.append(total_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"Episode: {episode}, Average Reward (last 100 episodes): {avg_reward:.2f}")

    print("Training completed.")
    return reward_history

def main():
    env = DualBrainSimpleEnv(render_mode="human")
    agent = DualBrainPolicyGradientAgent(env)
    reward_history = train(env, agent)

    plt.plot(reward_history)
    plt.title("Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig('dualbrain_rewards.png')

    # Test the trained agent
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
    print(f"Test episode total reward: {total_reward}")

if __name__ == "__main__":
    main()