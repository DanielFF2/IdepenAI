IndepenAI
----------------------------------------------------------------------------------------------------------------------------------
IndepenAI is a modular, easy-to-use reinforcement learning agent based on the Proximal Policy Optimization (PPO) algorithm. 
It is implemented in PyTorch and designed to work out-of-the-box with OpenAI Gym environments, 
supporting both continuous and discrete action spaces. 
IndepenAI is ideal for research, experimentation, and learning about modern RL techniques.

How does it work? (Architecture)
IndepenAI is built around three main components:

ActorNetwork:
This neural network decides which action to take given the current state. 
For continuous actions, it outputs the mean and standard deviation of a Gaussian distribution.

CriticNetwork:
This neural network estimates the value of a given state, helping the agent understand how good or bad a situation is.

PPOAgent:
This class manages the learning process. 
It collects experiences, calculates advantages, updates the actor and critic networks, and handles hyperparameters and memory.

The agent interacts with the environment, collects data, and periodically updates its networks using 
PPO’s clipped objective for stable and efficient learning.

What can be customized?

Network sizes:
You can change the number of layers or neurons in the Actor and Critic networks.

PPO hyperparameters:
Learning rate, discount factor (gamma), clipping epsilon, batch size, entropy coefficient, and number of update epochs can all be adjusted when creating the agent.

Environments:
IndepenAI works with any Gym-compatible environment (e.g., CartPole, BipedalWalker, LunarLander, etc.).

Checkpointing:
You can save and load model weights and optimizer states to resume training or deploy trained agents.

----------------------------------------------------------------------------------------------------------------------------------
How to use pg_agent (example BipedalWalker):

Remember you have to install all of the imports via your terminal, a guide is placed under this example

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import gym
    import os
    import json
    from pg_agent import PPOAgent
    
    if __name__ == "__main__":
        from pg_agent import PPOAgent
        import numpy as np

    #Only use this if you're having troble with compatibility in gym versions
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_ 

    import torch
    import matplotlib.pyplot as plt
    import gym
    import os
    import json
    
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

    num_envs = 16
    env = gym.vector.make("BipedalWalker-v3", num_envs=num_envs, asynchronous=False)
    state_size = env.single_observation_space.shape[0]
    action_size = env.single_action_space.shape[0]
    print("state_size:", state_size)
    print("state shape:", env.reset()[0].shape)

    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        lr=0.0003,
        gamma=0.99,
        clip_epsilon=0.2,
        update_epochs=6,
        batch_size=128,
        entropy_coef=0.05
    )

    num_episodes = 1000
    max_timesteps = 500
    episode_rewards = []
    starting_episode = 0

    #If training has already been done, this loads what the agent had already previously learned, so no need to start over! 
    checkpoint_path = "bipedalwalker_ppo_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        starting_episode = checkpoint['episode'] + 1
        episode_rewards = checkpoint['rewards']
        print(f"Loaded Checkpoint: continuing from episode {starting_episode}")
    else:
        print("No checkpoint found. Iniciating training.")

    agent.action_low = torch.tensor(env.single_action_space.low, device=agent.device)
    agent.action_high = torch.tensor(env.single_action_space.high, device=agent.device)

    for episode in range(starting_episode, starting_episode + num_episodes):
        agent.update_hyperparameters(episode - starting_episode, num_episodes)
        state = env.reset()[0]
        total_rewards = np.zeros(num_envs)
        
        for t in range(max_timesteps):
            action, log_prob = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, log_prob, reward, next_state, done)
            state = next_state
            total_rewards += reward  # Soma recompensa de cada ambiente

        agent.update()
        mean_reward = np.mean(total_rewards)
        episode_rewards.append(mean_reward)
        print(f"Episode {episode+1} | Reward: {mean_reward}")
        
        if (episode + 1) % 50 == 0:
            checkpoint = {
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
                'episode': episode,
                'rewards': episode_rewards
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved in episode {episode+1}")

    env.close()

    # Save checkpoint
    checkpoint = {
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
        'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
        'episode': starting_episode + num_episodes - 1,
        'rewards': episode_rewards
    }
    torch.save(checkpoint, checkpoint_path)
    print("Final checkpoint saved.")

    # Saving file after training is done
    torch.save(agent.actor.state_dict(), "bipedalwalker_ppo_actor.pth")
    torch.save(agent.critic.state_dict(), "bipedalwalker_ppo_critic.pth")

    # Plot the end result (many ways to do so)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
    plt.title("Reward per episode (BipedalWalker - PPO)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("bipedalwalker_ppo_rewards.png")
    plt.show()

----------------------------------------------------------------------------------------------------------------------------------

Installation Guide
1. Recommended Python Version
We recommend using Python 3.9 for best compatibility with Gym and reinforcement learning libraries.
If you use a different version, you may encounter issues installing some environments (especially those requiring native extensions like Box2D).
If you're using Windows, after installing python, dont forget to add it to your PATH in the environment variables.

2. Install Required Libraries
First, upgrade pip and wheel:

    pip install --upgrade pip wheel setuptools

Then, install the main dependencies:

    pip install torch numpy matplotlib gym[box2d] gym[all]  
    # [all] is optional, [box2d] is needed for BipedalWalker
    # Remember that you may need to install others if using other environments

If you encounter errors related to wheels (e.g., Box2D or MuJoCo), check your Python version and try to install other gym versions 
    (eg pip install gym==0.21.0 or pip install box2d-py or another version compatible with Python 3.9 )

3. Clone the repository

    git clone https://github.com/DanielFF2/IndepenAI.git

    cd IndepenAI

5. Run the Example

    python how_to_use_agents.py (put the example in a py file to run the code)
----------------------------------------------------------------------------------------------------------------------------------
If you want to use other environments, you need to change the environment name in gym.make(), 
and update the state_size and action_size according to the new environment’s observation and action spaces.

Recommend creating a environment.py and then using it in your main.py for bigger/independent environments.
----------------------------------------------------------------------------------------------------------------------------------
## Contributing

Contributions are welcome!
Please open an issue or submit a pull request if you have suggestions, bug fixes, or improvements.
Things from hyperparameters changes, added mechanics to the agent, 
anything to make it learn more efficiently is accepted with open arms!

## License

This project is independent (as the name sugests), so no issues with that!
