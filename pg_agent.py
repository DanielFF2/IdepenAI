import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_size)
        self.log_std = nn.Parameter(torch.ones(action_size) * -0.5)  # parâmetro treinável

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std

class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class PPOAgent:
    def __init__(self, state_size, action_size, lr=0.0003, gamma=0.99, clip_epsilon=0.2, update_epochs=8, batch_size=128, entropy_coef=0.05):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.initial_entropy_coef = entropy_coef
        self.min_entropy_coef = 0.01
        self.entropy_coef = entropy_coef  # Novo parâmetro para o entropy bonus

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_network = ActorNetwork(state_size, action_size).to(self.device)
        self.critic_network = CriticNetwork(state_size).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor_network.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic_network.parameters(), lr=lr)

        self.action_low = None
        self.action_high = None

        self.memory = []

    def update_hyperparameters(self, episode, total_episodes):
        # Atualiza entropy_coef dinamicamente para manter exploração inicial alta
        if total_episodes > 0:
            progress = episode / total_episodes
            # Decaimento gradual: alta exploração no início, refinamento no final
            self.entropy_coef = self.min_entropy_coef + (self.initial_entropy_coef - self.min_entropy_coef) * (1 - progress * 0.7)

    def remember(self, state, action, log_prob, reward, next_state, done):
        self.memory.append((state, action, log_prob, reward, next_state, done))

    def act(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        mean, std = self.actor_network(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        if hasattr(self, "action_low") and hasattr(self, "action_high"):
            action_clipped = torch.clamp(action, self.action_low, self.action_high)
        else:
            action_clipped = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        # Para ambientes paralelos, retorna arrays
        return action_clipped.cpu().numpy().squeeze(), log_prob.cpu().detach().numpy().squeeze()

    def compute_returns_and_advantages(self, rewards, dones, values, next_values):
        if isinstance(next_values, torch.Tensor):
            num_envs = next_values.shape[1] if next_values.dim() > 1 else 1
        else:
            num_envs = len(next_values[-1]) if hasattr(next_values[-1], "__len__") else 1
        
        gae_lambda = 0.95
        advantages = torch.zeros((len(rewards), num_envs), device=self.device)
        
        for env_idx in range(num_envs):
            last_gae = 0
            for t in reversed(range(len(rewards))):
                reward = rewards[t][env_idx] if isinstance(rewards[t], (list, np.ndarray, torch.Tensor)) else rewards[t]
                done = dones[t][env_idx] if isinstance(dones[t], (list, np.ndarray, torch.Tensor)) else dones[t]
                
                current_val = values[t][env_idx].item() if values.dim() > 1 else values[t].item()
                
                # CORREÇÃO CRÍTICA: próximo valor correto
                if t == len(rewards) - 1:
                    # Para o último timestep, usar bootstrap do estado final
                    next_val = next_values[-1][env_idx].item() if next_values.dim() > 1 else next_values[-1].item()
                else:
                    # Para timesteps intermediários, usar values do próximo timestep
                    next_val = values[t+1][env_idx].item() if values.dim() > 1 else values[t+1].item()
                
                delta = reward + self.gamma * next_val * (1-done) - current_val
                last_gae = delta + self.gamma * gae_lambda * (1-done) * last_gae
                advantages[t, env_idx] = last_gae
        
        # Simplificar: sempre reshape
        values_reshaped = values.view(len(rewards), num_envs)
        returns = advantages + values_reshaped.detach()
        
        # Normalizar vantagens
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

    def update(self):
        # Extrair e converter dados da memória
        states, actions, old_log_probs, rewards, next_states, dones = zip(*self.memory)
        states = np.array(states)         # (timesteps, num_envs, state_size)
        actions = np.array(actions)       # (timesteps, num_envs, action_size)
        old_log_probs = np.array(old_log_probs) # (timesteps, num_envs)
        rewards = np.array(rewards)       # (timesteps, num_envs)
        dones = np.array(dones)           # (timesteps, num_envs)
        next_states = np.array(next_states) # (timesteps, num_envs, state_size)
        
        if rewards.ndim == 1:
            rewards = rewards[:, None]
            dones = dones[:, None]

        # Flatten para (timesteps*num_envs, ...)
        num_steps, num_envs = rewards.shape
        states = states.reshape(-1, self.state_size)
        actions = actions.reshape(-1, self.action_size)
        old_log_probs = old_log_probs.reshape(-1, 1)
        rewards = rewards.reshape(num_steps, num_envs)
        dones = dones.reshape(num_steps, num_envs)
        next_states = next_states.reshape(-1, self.state_size)

        # Converter para tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # Calcular valores
        values = self.critic_network(states).view(num_steps, num_envs)
        next_values = self.critic_network(next_states).view(num_steps, num_envs)

        # Calcular returns e advantages (mantendo shape [num_steps, num_envs])
        returns, advantages = self.compute_returns_and_advantages(
            rewards, dones, values, next_values
        )

        # Flatten para batching
        returns = returns.view(-1, 1)
        advantages = advantages.view(-1, 1)
        
        # Número total de amostras
        dataset_size = states.shape[0]
        
        for _ in range(self.update_epochs):
            # Criar índices e embaralhar
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            # Processar mini-batches
            for start_idx in range(0, dataset_size, self.batch_size):
                # Selecionar índices para este mini-batch (cuidado com o último batch)
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Obter dados do mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass do ator
                mean, std = self.actor_network(batch_states)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                
                # Calcular ratio (π_θ / π_θ_old)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Entropy para exploração
                entropy = dist.entropy().mean()
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                # PPO loss com entropy bonus
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Critic loss
                values_pred = self.critic_network(batch_states)
                critic_loss = (batch_returns - values_pred).pow(2).mean()
                
                # Atualizar pesos
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=0.5)
                self.optimizer_actor.step()
                
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=0.5)
                self.optimizer_critic.step()
        
        self.memory = []