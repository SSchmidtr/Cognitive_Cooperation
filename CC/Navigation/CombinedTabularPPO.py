import numpy as np
import matplotlib.pyplot as plt
import pickle
from combined_envV2 import CombinedEnv

class BrainPolicy:
    def __init__(self, action_space, learning_rate=0.01, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.policy = {}
        self.loss_history = []

    def get_action(self, state):
        state_key = tuple(state.flatten())
        if state_key not in self.policy:
            self.policy[state_key] = np.zeros(self.action_space.n)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space.n)
        else:
            probs = self.softmax(self.policy[state_key])
            return np.random.choice(self.action_space.n, p=probs)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def update_policy(self, episode):
        discounted_rewards = []
        for t in range(len(episode)):
            Gt = sum([r * (self.gamma ** i) for i, (_, _, r) in enumerate(episode[t:])])
            discounted_rewards.append(Gt)
        discounted_rewards = np.array(discounted_rewards)

        baseline = np.mean(discounted_rewards)
        advantages = discounted_rewards - baseline
        loss = 0

        for t, (state, action, _) in enumerate(episode):
            state_key = tuple(state.flatten())
            if state_key not in self.policy:
                self.policy[state_key] = np.zeros(self.action_space.n)

            probs = self.softmax(self.policy[state_key])
            grad = -probs
            grad[action] += 1

            loss -= np.log(probs[action]) * advantages[t]
            self.policy[state_key] += self.learning_rate * advantages[t] * grad

        self.loss_history.append(loss)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_policy(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.policy, f)

    def plot_loss(self):
        plt.figure()
        plt.plot(self.loss_history)
        plt.title("Policy Loss over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.show()

class CombinedAgent:
    def __init__(self, env):
        self.env = env
        self.brain1 = BrainPolicy(env.brain1_action_space)
        self.brain2 = BrainPolicy(env.brain2_action_space)

    def get_combined_action(self, state):
        action1 = self.brain1.get_action(state['brain1'])
        action2 = self.brain2.get_action(state['brain2'])
        combined_action = self.env.combined_actions.index((action1, action2))
        return combined_action

    def train(self, num_episodes):
        reward_history = []

        for episode in range(num_episodes):
            obs, info = self.env.reset()
            state = obs
            done = False
            episode_memory1 = []
            episode_memory2 = []
            total_reward = 0

            while not done:
                combined_action = self.get_combined_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(combined_action)
                next_state = next_obs
                done = terminated or truncated

                brain1_action, brain2_action = self.env.combined_actions[combined_action]
                episode_memory1.append((state['brain1'], brain1_action, reward))
                episode_memory2.append((state['brain2'], brain2_action, reward))

                state = next_state
                total_reward += reward

            self.brain1.update_policy(episode_memory1)
            self.brain2.update_policy(episode_memory2)
            reward_history.append(total_reward)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        self.brain1.save_policy('brain1_policy.pkl')
        self.brain2.save_policy('brain2_policy.pkl')
        self.brain1.plot_loss()
        self.brain2.plot_loss()

        return reward_history

def main():
    env = CombinedEnv(render_mode=None)
    agent = CombinedAgent(env)
    num_episodes = 10000
    reward_history = agent.train(num_episodes=num_episodes)

    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards over Episodes")
    plt.show()

if __name__ == "__main__":
    main()
