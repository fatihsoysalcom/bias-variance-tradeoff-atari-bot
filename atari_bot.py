import gym
import numpy as np

# This is a simplified demonstration. A real DRL agent would use neural networks.
# The core idea here is to illustrate the bias-variance tradeoff in a conceptual way.

class SimpleAgent:
    def __init__(self, action_space_size, initial_bias=0.5, initial_variance=0.1):
        self.action_space_size = action_space_size
        self.bias = initial_bias  # Represents systematic error (high bias = underfitting)
        self.variance = initial_variance # Represents sensitivity to training data (high variance = overfitting)

    def predict_action(self, observation):
        # In a real DRL agent, this would be a complex function (e.g., a neural network).
        # Here, we simulate a prediction with noise based on our bias and variance.
        # A high bias means the prediction is consistently off.
        # A high variance means the prediction can jump around a lot.
        base_prediction = np.random.uniform(0, self.action_space_size - 1)
        noisy_prediction = base_prediction + np.random.normal(loc=self.bias, scale=self.variance)
        
        # Clamp prediction to valid action range
        return np.clip(int(noisy_prediction), 0, self.action_space_size - 1)

    def adjust_parameters(self, performance_metric):
        # This is a highly simplified adjustment. In DRL, this involves gradient descent.
        # If performance is poor (e.g., low reward), we might adjust bias/variance.
        # For demonstration, let's say higher performance allows more exploration (higher variance)
        # and lower performance requires more caution (lower variance).
        
        # Example: If performance is good, we might slightly increase variance to explore more.
        # If performance is bad, we might decrease variance to stabilize.
        if performance_metric > 0.5: # Assume 0.5 is a threshold for 'good' performance
            self.variance = min(0.5, self.variance * 1.05) # Increase variance slightly
            self.bias = max(0.0, self.bias * 0.95) # Decrease bias slightly to refine
        else:
            self.variance = max(0.01, self.variance * 0.95) # Decrease variance to stabilize
            self.bias = min(1.0, self.bias * 1.05) # Increase bias slightly if stuck

        print(f"Adjusted: Bias={self.bias:.3f}, Variance={self.variance:.3f}")


def train_agent(agent, env, episodes=100):
    for episode in range(episodes):
        observation = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.predict_action(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward

        # In a real DRL scenario, the agent would learn from the reward.
        # Here, we use total_reward as a proxy for performance to adjust parameters.
        agent.adjust_parameters(total_reward / 100.0) # Normalize reward for adjustment
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    # Using a simple environment for demonstration
    # A real Atari environment would be much more complex.
    # For this example, we'll use CartPole, which is simpler but still has state.
    try:
        env = gym.make('CartPole-v1')
    except gym.error.DependencyNotInstalled:
        print("Gymnasium not installed. Please install it: pip install gymnasium")
        exit()
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Ensure you have gymnasium installed: pip install gymnasium")
        exit()

    action_space_size = env.action_space.n
    # Initialize agent with moderate bias and variance
    agent = SimpleAgent(action_space_size, initial_bias=0.3, initial_variance=0.2)

    print("Starting training...")
    train_agent(agent, env, episodes=50)
    print("Training finished.")

    # Demonstrate final prediction after training
    print("\nDemonstrating prediction after training:")
    sample_observation = env.observation_space.sample()
    final_action = agent.predict_action(sample_observation)
    print(f"Sample Observation: {sample_observation}")
    print(f"Predicted Action: {final_action}")
    print(f"Final Agent Parameters: Bias={agent.bias:.3f}, Variance={agent.variance:.3f}")

    env.close()
