import numpy as np
import matplotlib.pyplot as plt
import pickle
from combined_envV2 import CombinedEnv
from numba import njit

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
def update_policy_numba(policy_logits, actions, old_log_probs, returns, advantages, learning_rate, epsilon, value_function_values):
    loss_history = []
    for i in range(len(actions)):
        logits = policy_logits[i]
        probs = softmax(logits)
        new_prob = probs[actions[i]]
        old_prob = np.exp(old_log_probs[i])
        ratio = new_prob / old_prob

        # Clipping para PPO
        if ((advantages[i] >= 0 and ratio <= 1 + epsilon) or
            (advantages[i] < 0 and ratio >= 1 - epsilon)):
            surr = ratio * advantages[i]
            actor_loss = -surr

            # Gradiente y actualización
            one_hot = np.zeros_like(probs)
            one_hot[actions[i]] = 1
            delta_vector = one_hot - probs
            grad = (advantages[i] / old_prob) * new_prob * delta_vector
            policy_logits[i] += learning_rate * grad

            # Actualizar función de valor
            value_function_values[i] += learning_rate * (returns[i] - value_function_values[i])

            loss_history.append(actor_loss)
        else:
            loss_history.append(0.0)
    return policy_logits, value_function_values, loss_history

class BrainPolicy:
    def __init__(self, action_space, learning_rate=0.01, gamma=0.99, epsilon=0.2):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon  # Factor de clipping para PPO
        self.policy = {}
        self.value_function = {}
        self.loss_history = []

        # Diccionarios para almacenar las frecuencias y probabilidades de las acciones del otro agente
        self.other_agent_action_counts = {}
        self.other_agent_action_probs = {}

    def get_action(self, state, predicted_other_action=None):
        state_key = tuple(state.flatten())
        if state_key not in self.policy:
            self.policy[state_key] = np.zeros(self.action_space.n)
            self.value_function[state_key] = 0.0

        logits = self.policy[state_key]
        probs = softmax(logits)

        # Si hay una predicción, ajustamos las probabilidades
        if predicted_other_action is not None:
            probs = self.adjust_probs_for_prediction(probs, predicted_other_action)

        # Normalizamos las probabilidades
        probs = probs / probs.sum()

        # Validación: aseguramos que las probabilidades sumen 1
        if not np.isclose(probs.sum(), 1):
            raise ValueError("Las probabilidades no suman 1 después de la normalización.")

        action = np.random.choice(self.action_space.n, p=probs)
        log_prob = np.log(probs[action])

        return action, log_prob, self.value_function[state_key]

    def adjust_probs_for_prediction(self, probs, predicted_action):
        adjusted_probs = np.copy(probs)
        adjusted_probs[predicted_action] += 0.1  # Fomentamos la acción predicha
        adjusted_probs = np.clip(adjusted_probs, 0, None)
        adjusted_probs /= adjusted_probs.sum()
        return adjusted_probs

    def predict_other_action(self, other_state):
        state_key = tuple(other_state.flatten())
        if state_key in self.other_agent_action_probs:
            action_probs = self.other_agent_action_probs[state_key]
            predicted_action = np.random.choice(self.action_space.n, p=action_probs)
        else:
            # Si no tenemos datos, elegimos una acción aleatoria
            predicted_action = np.random.randint(0, self.action_space.n)
        return predicted_action

    def update_other_agent_model(self):
        # Actualizamos las probabilidades basadas en los recuentos
        for state_key, action_counts in self.other_agent_action_counts.items():
            total_counts = np.sum(action_counts)
            if total_counts > 0:
                self.other_agent_action_probs[state_key] = action_counts / total_counts
            else:
                # Si no se han observado acciones, asignamos probabilidades uniformes
                self.other_agent_action_probs[state_key] = np.ones(self.action_space.n) / self.action_space.n

        # Limpiamos los recuentos para evitar problemas de memoria
        self.other_agent_action_counts = {}

    def record_other_agent_action(self, other_state, other_action):
        state_key = tuple(other_state.flatten())
        if state_key not in self.other_agent_action_counts:
            self.other_agent_action_counts[state_key] = np.zeros(self.action_space.n)
        self.other_agent_action_counts[state_key][other_action] += 1

    def update_policy(self, states, actions, old_log_probs, returns, advantages):
        # Preparamos los datos para Numba
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

        # Actualizamos la política y la función de valor usando Numba
        updated_logits, updated_values, loss_history = update_policy_numba(
            policy_logits, actions, old_log_probs, returns, advantages,
            self.learning_rate, self.epsilon, value_function_values
        )

        # Actualizamos los diccionarios con los valores actualizados
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
        plt.title("Pérdida de la Política a lo largo de los Episodios")
        plt.xlabel("Episodio")
        plt.ylabel("Pérdida")
        plt.show()

class CombinedAgent:
    def __init__(self, env):
        self.env = env
        self.brain1 = BrainPolicy(env.brain1_action_space)
        self.brain2 = BrainPolicy(env.brain2_action_space)

    def get_combined_action(self, state):
        predicted_action_brain2 = self.brain2.predict_other_action(state['brain2'])
        brain_action1, log_prob1, value1 = self.brain1.get_action(state['brain1'], predicted_action_brain2)

        predicted_action_brain1 = self.brain1.predict_other_action(state['brain1'])
        brain_action2, log_prob2, value2 = self.brain2.get_action(state['brain2'], predicted_action_brain1)

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
                next_obs, reward, terminated, truncated, _ = self.env.step(combined_action)
                next_state = next_obs
                done = terminated or truncated

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

                # Registramos la acción del otro agente para actualizar el modelo
                self.brain1.record_other_agent_action(state['brain1'], brain2_action)
                self.brain2.record_other_agent_action(state['brain2'], brain1_action)

                state = next_state

            # Actualizamos los modelos del otro agente
            self.brain1.update_other_agent_model()
            self.brain2.update_other_agent_model()

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
            print(f"Episodio {episode + 1}/{num_episodes}, Recompensa Total: {total_reward}")

        self.brain1.save_policy('brain1_policy.pkl')
        self.brain2.save_policy('brain2_policy.pkl')
        self.plot_trend_line(reward_history)
        return reward_history

    def plot_trend_line(self, rewards):
        plt.figure()
        plt.plot(rewards, label="Recompensa")
        window_size = 100
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
            plt.plot(range(len(moving_avg)), moving_avg, color='red', label="Tendencia (Media Móvil)")
        plt.title("Recompensas a lo largo de los Episodios de Entrenamiento con Tendencia")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa Total")
        plt.legend()
        plt.show()

    @staticmethod
    def main():
        env = CombinedEnv()
        agent = CombinedAgent(env)
        num_episodes = 5000
        reward_history = agent.train(num_episodes=num_episodes)

if __name__ == "__main__":
    CombinedAgent.main()
