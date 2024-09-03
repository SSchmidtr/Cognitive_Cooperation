# Import necessary libraries
import numpy as np  # For numerical operations
from simplegridv2 import SimpleEnv  # Import the custom environment
import matplotlib.pyplot as plt  # For plotting

class PolicyGradientAgent:
    """
    A class implementing a Policy Gradient agent for reinforcement learning.
    This agent learns a policy to maximize expected rewards in an environment.
    """

    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        """
        Initialize the Policy Gradient agent.

        Args:
        env: The environment the agent will interact with
        learning_rate: The rate at which the agent updates its policy
        gamma: The discount factor for future rewards
        """
        self.env = env  # Store the environment
        self.learning_rate = learning_rate  # Set the learning rate
        self.gamma = gamma  # Set the discount factor
        self.state_size = env.observation_space['image'].shape  # Get the shape of the state space
        self.action_size = env.action_space.n  # Get the number of possible actions
        self.policy = {}  # Initialize an empty dictionary to store the policy

    def get_action(self, state):
        """
        Choose an action based on the current state using the learned policy.

        Args:
        state: The current state of the environment

        Returns:
        action: The chosen action
        """
        state_key = tuple(state.flatten())  # Convert the state to a tuple for dictionary key
        if state_key not in self.policy:
            # If this state hasn't been seen before, initialize a random policy for it
            self.policy[state_key] = np.random.rand(self.action_size)
        # Calculate action probabilities using softmax
        probs = np.exp(self.policy[state_key]) / np.sum(np.exp(self.policy[state_key]))
        # Choose an action probabilistically
        return np.random.choice(self.action_size, p=probs)

    def update_policy(self, episode):
        """
        Update the policy based on the episode's outcomes.

        Args:
        episode: A list of (state, action, reward) tuples from the episode
        """
        discounted_rewards = []
        for t in range(len(episode)):
            Gt = 0  # Initialize cumulative reward
            pw = 0  # Initialize power for gamma
            for r in episode[t:]:
                Gt = Gt + self.gamma**pw * r[2]  # Calculate discounted reward
                pw = pw + 1  # Increment power
            discounted_rewards.append(Gt)  # Add cumulative discounted reward to list

        # Convert to numpy array and normalize
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)

        # Update policy for each step in the episode
        for t, (state, action, reward) in enumerate(episode):
            state_key = tuple(state.flatten())  # Convert state to tuple for dictionary key
            if state_key not in self.policy:
                # If this state hasn't been seen before, initialize a random policy for it
                self.policy[state_key] = np.random.rand(self.action_size)
            # Calculate action probabilities using softmax
            probs = np.exp(self.policy[state_key]) / np.sum(np.exp(self.policy[state_key]))
            # Calculate the gradient
            grad = np.zeros(self.action_size)
            grad[action] = 1 - probs[action]
            # Update the policy
            self.policy[state_key] += self.learning_rate * discounted_rewards[t] * grad

def train(env, agent, num_episodes=1000):
    """
    Train the agent in the given environment.

    Args:
    env: The environment to train in
    agent: The agent to train
    num_episodes: The number of episodes to train for
    """
    for episode in range(num_episodes):
        state, _ = env.reset()  # Reset the environment
        done = False
        episode_memory = []
        while not done:
            action = agent.get_action(state['image'])  # Choose an action
            next_state, reward, terminated, truncated, _ = env.step(action)  # Take the action
            done = terminated or truncated  # Check if episode is over
            episode_memory.append((state['image'], action, reward))  # Store the transition
            state = next_state  # Move to the next state
        agent.update_policy(episode_memory)  # Update the policy after the episode
        print(f"Episode: {episode}, Reward: {sum([r[2] for r in episode_memory])}")
        reward_grow = []

        if episode % 100 == 0:
            print(f"Episode: {episode}, Reward: {sum([r[2] for r in episode_memory])}")
        
        if episode in range(900,1000):
            average_reward = sum([r[2] for r in episode_memory])/100
            print(f"Episode: {episode}, Average Reward: {average_reward}")

        if episode in range(1000):
            average_reward = sum([r[2] for r in episode_memory])/100
            reward_grow.append(average_reward)


    print("Training completed.")

def main():
    """
    Main function to run the training and testing of the agent.
    """
    env = SimpleEnv(render_mode="human")  # Create the environment
    agent = PolicyGradientAgent(env)  # Create the agent
    train(env, agent)  # Train the agent

    # Test the trained agent
    state, _ = env.reset()  # Reset the environment
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state['image'])  # Choose an action
        state, reward, terminated, truncated, _ = env.step(action)  # Take the action
        done = terminated or truncated  # Check if episode is over
        total_reward += reward  # Accumulate reward
        env.render()  # Render the environment
    print(f"Total reward: {total_reward}")  # Print the total reward

if __name__ == "__main__":
    main()  # Run the main function if this script is executed