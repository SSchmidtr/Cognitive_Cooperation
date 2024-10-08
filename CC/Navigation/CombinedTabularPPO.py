import numpy as np
import matplotlib.pyplot as plt
import pickle
from combined_envV2 import CombinedEnv

class BrainPolicy:
    def __init__(self, action_space, learning_rate=0.01, gamma=0.99, epsilon=0.2):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon  # Clipping factor for PPO
        self.policy = {}
        self.value_function = {}
        self.loss_history = []

    def get_action(self, state):
        state_key = tuple(state.flatten())
        if state_key not in self.policy:
            self.policy[state_key] = np.zeros(self.action_space.n)
            self.value_function[state_key] = 0.0

        # Calculamos las probabilidades de las acciones usando softmax
        probs = self.softmax(self.policy[state_key])
        action = np.random.choice(self.action_space.n, p=probs)
        log_prob = np.log(probs[action])

        return action, log_prob, self.value_function[state_key]

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def compute_advantages(self, rewards, values, next_value):
        returns = []
        Gt = next_value
        for r in reversed(rewards):
            Gt = r + self.gamma * Gt
            returns.insert(0, Gt)
        returns = np.array(returns)
        advantages = returns - values
        return advantages, returns

    def update_policy(self, states, actions, old_log_probs, returns, advantages):
        for i, state in enumerate(states):
            state_key = tuple(state.flatten())
            if state_key not in self.policy:
                self.policy[state_key] = np.zeros(self.action_space.n)
                self.value_function[state_key] = 0.0

            probs = self.softmax(self.policy[state_key])
            new_log_prob = np.log(probs[actions[i]])
            ratio = np.exp(new_log_prob - old_log_probs[i])

            # PPO Clipping
            surr1 = ratio * advantages[i]
            surr2 = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages[i]
            actor_loss = -np.minimum(surr1, surr2)

            # Update policy and value function
            self.policy[state_key] += self.learning_rate * actor_loss
            self.value_function[state_key] += self.learning_rate * (returns[i] - self.value_function[state_key])

        # Guardamos el loss acumulado
        self.loss_history.append(actor_loss.sum())

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
        action1, log_prob1, value1 = self.brain1.get_action(state['brain1'])
        action2, log_prob2, value2 = self.brain2.get_action(state['brain2'])
        combined_action = self.env.combined_actions.index((action1, action2))
        return combined_action, log_prob1, log_prob2, value1, value2

    def train(self, num_episodes):
        reward_history = []

        for episode in range(num_episodes):
            obs, info = self.env.reset()
            state = obs
            done = False
            episode_memory1 = []
            episode_memory2 = []
            total_reward = 0
            rewards = []
            values_brain1 = []
            values_brain2 = []
            log_probs_brain1 = []
            log_probs_brain2 = []

            while not done:
                combined_action, log_prob1, log_prob2, value1, value2 = self.get_combined_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(combined_action)
                next_state = next_obs
                done = terminated or truncated

                brain1_action, brain2_action = self.env.combined_actions[combined_action]

                # Suponiendo que el reward es un tuple (reward_brain1, reward_brain2)
                reward_brain1, reward_brain2 = reward  # Desempaquetar las recompensas de cada cerebro
                total_reward += reward_brain1 + reward_brain2  # Sumar las recompensas de ambos cerebros

                # Guardar datos para actualizar la política
                episode_memory1.append((state['brain1'], brain1_action))
                episode_memory2.append((state['brain2'], brain2_action))
                rewards.append(reward_brain1 + reward_brain2)  # Sumar las recompensas para el cálculo de ventajas
                log_probs_brain1.append(log_prob1)
                log_probs_brain2.append(log_prob2)
                values_brain1.append(value1)
                values_brain2.append(value2)

                state = next_state

            # Al final del episodio, calculamos las ventajas y actualizamos la política
            next_value_brain1 = self.brain1.value_function[tuple(next_state['brain1'].flatten())] if not done else 0
            next_value_brain2 = self.brain2.value_function[tuple(next_state['brain2'].flatten())] if not done else 0

            advantages_brain1, returns_brain1 = self.brain1.compute_advantages(rewards, values_brain1, next_value_brain1)
            advantages_brain2, returns_brain2 = self.brain2.compute_advantages(rewards, values_brain2, next_value_brain2)

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

    # Función para graficar la tendencia de recompensas durante los episodios
    def plot_trend_line(self, rewards):
        plt.figure()
        plt.plot(rewards, label="Reward")  # Graficamos las recompensas

        # Calculamos una línea de tendencia usando el promedio móvil
        window_size = 50  # Tamaño de la ventana para el promedio móvil
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

        plt.plot(range(len(moving_avg)), moving_avg, color='red', label="Trend (Moving Average)")
        plt.title("Rewards over Training Episodes with Trend")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.show()

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
    CombinedAgent.main()
