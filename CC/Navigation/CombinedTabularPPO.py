import numpy as np
import matplotlib.pyplot as plt
import pickle
from combined_envV2 import CombinedEnv
from numba import njit, prange

# Funciones auxiliares compiladas con Numba
@njit
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

@njit
def compute_advantages(rewards, values, next_value, gamma):
    returns = np.zeros(len(rewards))
    Gt = next_value
    for t in range(len(rewards) - 1, -1, -1):
        Gt = rewards[t] + gamma * Gt
        returns[t] = Gt
    advantages = returns - values
    return advantages, returns

@njit
def update_policy_numba(policy_logits, actions, old_log_probs, returns, advantages, learning_rate, epsilon, value_function_values, value_learning_rate, max_inner_iterations):
    loss_history = []
    for i in range(len(actions)):
        inner_iteration = 0
        while True:
            logits = policy_logits[i]
            probs = softmax(logits)
            new_prob = probs[actions[i]]
            old_prob = np.exp(old_log_probs[i])
            ratio = new_prob / old_prob

            if ((advantages[i] >= 0 and ratio <= 1 + epsilon) or
                (advantages[i] < 0 and ratio >= 1 - epsilon)):
                surr = ratio * advantages[i]
                actor_loss = -surr

                one_hot = np.zeros_like(probs)
                one_hot[actions[i]] = 1
                delta_vector = one_hot - probs
                grad = (advantages[i] / old_prob) * new_prob * delta_vector

                policy_logits[i] += learning_rate * grad

                value_function_values[i] += value_learning_rate * (returns[i] - value_function_values[i])

                inner_iteration += 1
                if inner_iteration >= max_inner_iterations:
                    break
            else:
                break
        loss_history.append(actor_loss)
    return policy_logits, value_function_values, loss_history

class BrainPolicy:
    def __init__(self, action_space, learning_rate=0.01, value_learning_rate=0.1, gamma=0.99, epsilon=0.2):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.value_learning_rate = value_learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = {}  # Diccionario para almacenar los logits de la política
        self.value_function = {}  # Diccionario para almacenar los valores de estado
        self.loss_history = []

    def get_action(self, state):
        state_key = tuple(state.flatten())
        if state_key not in self.policy:
            self.policy[state_key] = np.zeros(self.action_space.n)
            self.value_function[state_key] = 0.0

        logits = self.policy[state_key]
        probs = softmax(logits)
        action = np.random.choice(self.action_space.n, p=probs)
        log_prob = np.log(probs[action])

        return action, log_prob, self.value_function[state_key]

    def update_policy(self, states, actions, old_log_probs, returns, advantages):
        max_inner_iterations = 10

        # Preparar datos para Numba
        policy_logits = []
        value_function_values = []
        for state in states:
            state_key = tuple(state.flatten())
            policy_logits.append(self.policy[state_key])
            value_function_values.append(self.value_function[state_key])

        policy_logits = np.array(policy_logits)
        value_function_values = np.array(value_function_values)
        actions = np.array(actions)
        old_log_probs = np.array(old_log_probs)
        returns = np.array(returns)
        advantages = np.array(advantages)

        # Actualizar política y valor usando Numba
        updated_logits, updated_values, loss_history = update_policy_numba(
            policy_logits, actions, old_log_probs, returns, advantages,
            self.learning_rate, self.epsilon, value_function_values,
            self.value_learning_rate, max_inner_iterations
        )

        # Actualizar los diccionarios con los valores actualizados
        for i, state in enumerate(states):
            state_key = tuple(state.flatten())
            self.policy[state_key] = updated_logits[i]
            self.value_function[state_key] = updated_values[i]
            self.loss_history.append(loss_history[i])

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
        self.brain1 = BrainPolicy(env.brain1_action_space, learning_rate=0.01, value_learning_rate=0.1)
        self.brain2 = BrainPolicy(env.brain2_action_space, learning_rate=0.01, value_learning_rate=0.1)

    def get_combined_action(self, state):
        brain_action1, log_prob1, value1 = self.brain1.get_action(state['brain1'])
        brain_action2, log_prob2, value2 = self.brain2.get_action(state['brain2'])

        combined_action = self.env.combined_actions.index((brain_action1, brain_action2))
        return combined_action, log_prob1, log_prob2, value1, value2, brain_action1, brain_action2

    def train(self, num_episodes):
        reward_history = []

        for episode in range(num_episodes):
            obs, info = self.env.reset()
            state = obs
            done = False
            episode_memory1 = []
            episode_memory2 = []
            total_reward = 0
            values_brain1 = []
            values_brain2 = []
            log_probs_brain1 = []
            log_probs_brain2 = []
            rewards_brain1 = []
            rewards_brain2 = []

            while not done:
                combined_action, log_prob1, log_prob2, value1, value2, brain1_action, brain2_action = self.get_combined_action(state)
                print(f"Episode: {episode + 1}, Action: {combined_action}, Brain 1 Action: {brain1_action}, Brain 2 Action: {brain2_action}")
                next_obs, reward, terminated, truncated, _ = self.env.step(combined_action)
                next_state = next_obs
                done = terminated or truncated

                brain1_action, brain2_action = self.env.combined_actions[combined_action]

                reward_brain1, reward_brain2 = reward
                total_reward += reward_brain1 + reward_brain2

                rewards_brain1.append(reward_brain1)
                rewards_brain2.append(reward_brain2)

                episode_memory1.append((state['brain1'], brain1_action))
                episode_memory2.append((state['brain2'], brain2_action))
                log_probs_brain1.append(log_prob1)
                log_probs_brain2.append(log_prob2)
                values_brain1.append(value1)
                values_brain2.append(value2)

                state = next_state

            next_state_brain1 = next_state['brain1']
            next_state_brain2 = next_state['brain2']

            next_value_brain1 = self.brain1.value_function.get(tuple(next_state_brain1.flatten()), 0)
            next_value_brain2 = self.brain2.value_function.get(tuple(next_state_brain2.flatten()), 0)

            advantages_brain1, returns_brain1 = compute_advantages(
                np.array(rewards_brain1), np.array(values_brain1), next_value_brain1, self.brain1.gamma
            )
            advantages_brain2, returns_brain2 = compute_advantages(
                np.array(rewards_brain2), np.array(values_brain2), next_value_brain2, self.brain2.gamma
            )

            self.brain1.update_policy(
                [mem[0] for mem in episode_memory1],
                [mem[1] for mem in episode_memory1],
                log_probs_brain1,
                returns_brain1,
                advantages_brain1
            )
            self.brain2.update_policy(
                [mem[0] for mem in episode_memory2],
                [mem[1] for mem in episode_memory2],
                log_probs_brain2,
                returns_brain2,
                advantages_brain2
            )

            reward_history.append(total_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        self.brain1.save_policy('brain1_policy.pkl')
        self.brain2.save_policy('brain2_policy.pkl')
        self.plot_trend_line(reward_history)
        return reward_history

    def plot_trend_line(self, rewards):
        plt.figure()
        plt.plot(rewards, label="Reward")
        window_size = 100
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
            plt.plot(range(len(moving_avg)), moving_avg, color='red', label="Trend (Moving Average)")
        plt.title("Rewards over Training Episodes with Trend")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.show()

    @staticmethod
    def main():
        env = CombinedEnv()
        agent = CombinedAgent(env)
        num_episodes = 3000
        reward_history = agent.train(num_episodes=num_episodes)

if __name__ == "__main__":
    CombinedAgent.main()
