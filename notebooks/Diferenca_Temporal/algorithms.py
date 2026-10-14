"""
algorithms.py
=============

Implementações de algoritmos de Aprendizado por Reforço.

Algoritmos de Predição (Avaliação de Política):
------------------------------------------------
- td_zero_prediction(): TD(0) para avaliação de política
- first_visit_mc_prediction(): Monte Carlo First-Visit para avaliação

Algoritmos de Controle (Busca de Política Ótima):
--------------------------------------------------
- sarsa(): SARSA (on-policy TD control)
- q_learning(): Q-Learning (off-policy TD control)
- expected_sarsa(): Expected SARSA (off-policy TD control)
- first_visit_mc_control(): Monte Carlo First-Visit Control
- mc_exploring_starts(): Monte Carlo com Exploring Starts

Funções Auxiliares:
-------------------
- get_greedy_policy(): Extrai política gulosa de Q-values
- epsilon_greedy_action(): Escolhe ação usando ε-greedy

Autor: Material Educacional RL
Data: 2025
"""

import numpy as np
from typing import Tuple, Dict, List
from environment import GridWorld


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def state_to_idx(state: Tuple[int, int], n_cols: int) -> int:
    """
    Converte estado (row, col) para índice linear.
    
    Parâmetros:
    -----------
    state : Tuple[int, int]
        Estado (row, col)
    n_cols : int
        Número de colunas do grid
    
    Retorna:
    --------
    int
        Índice linear
    """
    return state[0] * n_cols + state[1]


def epsilon_greedy_action(Q: np.ndarray, state: Tuple[int, int],
                         gridworld: GridWorld, epsilon: float) -> int:
    """
    Escolhe ação usando política ε-greedy.
    
    Com probabilidade ε: explora (ação aleatória)
    Com probabilidade (1-ε): exploita (melhor ação)
    
    Parâmetros:
    -----------
    Q : np.ndarray
        Tabela Q de shape (n_states, n_actions)
    state : Tuple[int, int]
        Estado atual
    gridworld : GridWorld
        Ambiente
    epsilon : float
        Probabilidade de exploração
    
    Retorna:
    --------
    int
        Índice da ação escolhida
    """
    if np.random.random() < epsilon:
        return np.random.randint(len(gridworld.actions))
    else:
        state_idx = state_to_idx(state, gridworld.cols)
        return np.argmax(Q[state_idx])


def get_greedy_policy(Q: np.ndarray, gridworld: GridWorld) -> Dict:
    """
    Extrai política gulosa dos Q-values.
    
    Para cada estado, escolhe a ação com maior Q(s,a).
    
    Parâmetros:
    -----------
    Q : np.ndarray
        Tabela Q
    gridworld : GridWorld
        Ambiente
    
    Retorna:
    --------
    Dict
        Política {estado: ação}
    """
    policy = {}
    
    for state in gridworld.states:
        if not gridworld.is_terminal(state):
            state_idx = state_to_idx(state, gridworld.cols)
            best_action_idx = np.argmax(Q[state_idx])
            policy[state] = gridworld.actions[best_action_idx]
    
    return policy


# ============================================================================
# ALGORITMOS DE PREDIÇÃO (AVALIAÇÃO DE POLÍTICA)
# ============================================================================

def td_zero_prediction(gridworld: GridWorld, policy: Dict,
                       n_episodes: int = 1000, alpha: float = 0.1,
                       initial_state: Tuple[int, int] = None,
                       verbose: bool = False) -> Dict:
    """
    Algoritmo TD(0) para avaliação de política.
    
    Estima V^π(s) - o valor de cada estado seguindo política π.
    Atualiza V(s) a cada passo usando diferença temporal.
    
    Fórmula:
    --------
    V(St) ← V(St) + α[Rt+1 + γV(St+1) - V(St)]
    
    Parâmetros:
    -----------
    gridworld : GridWorld
        Ambiente
    policy : Dict
        Política a avaliar {estado: ação}
    n_episodes : int, default=1000
        Número de episódios de treinamento
    alpha : float, default=0.1
        Taxa de aprendizado (0 < α ≤ 1)
        - Controla quanto aprendemos com cada experiência
        - Valores típicos: 0.01 a 0.5
    initial_state : Tuple[int, int], opcional
        Estado inicial (se None, escolhe aleatoriamente)
    verbose : bool, default=False
        Se True, imprime progresso
    
    Retorna:
    --------
    Dict
        Função valor V {estado: valor}
    
    Exemplo:
    --------
    >>> from environment import create_classic_gridworld
    >>> gw = create_classic_gridworld()
    >>> policy = {s: 'N' for s in gw.states if not gw.is_terminal(s)}
    >>> V = td_zero_prediction(gw, policy, n_episodes=1000, alpha=0.1)
    """
    # Inicializa V(s) = 0
    V = {state: 0.0 for state in gridworld.states}
    
    # Estados terminais têm V = 0
    for terminal_state in gridworld.terminal_states:
        V[terminal_state] = 0.0
    
    gamma = gridworld.gamma
    
    for episode in range(n_episodes):
        # Estado inicial
        if initial_state is not None:
            state = initial_state
        else:
            non_terminal = [s for s in gridworld.states
                           if not gridworld.is_terminal(s)]
            state = non_terminal[np.random.randint(len(non_terminal))]
        
        # Executa episódio
        while not gridworld.is_terminal(state):
            action = policy[state]
            next_state, reward = gridworld.sample_transition(state, action)
            
            # Atualização TD(0)
            td_target = reward + gamma * V[next_state]
            td_error = td_target - V[state]
            V[state] = V[state] + alpha * td_error
            
            state = next_state
        
        if verbose and (episode + 1) % 100 == 0:
            avg_value = np.mean([V[s] for s in gridworld.states
                                if not gridworld.is_terminal(s)])
            print(f"Episódio {episode + 1}/{n_episodes} - V médio: {avg_value:.4f}")
    
    return V


def first_visit_mc_prediction(gridworld: GridWorld, policy: Dict,
                              n_episodes: int = 1000, alpha: float = 0.1,
                              initial_state: Tuple[int, int] = None,
                              verbose: bool = False) -> Dict:
    """
    Monte Carlo First-Visit para avaliação de política.
    
    Estima V^π(s) usando retornos completos dos episódios.
    Atualiza V(s) apenas na primeira visita a s em cada episódio.
    
    Fórmula:
    --------
    V(s) ← V(s) + α[Gt - V(s)]
    onde Gt = Rt+1 + γRt+2 + γ²Rt+3 + ...
    
    Parâmetros:
    -----------
    gridworld : GridWorld
        Ambiente
    policy : Dict
        Política a avaliar
    n_episodes : int, default=1000
        Número de episódios
    alpha : float, default=0.1
        Taxa de aprendizado
    initial_state : Tuple[int, int], opcional
        Estado inicial
    verbose : bool, default=False
        Se True, imprime progresso
    
    Retorna:
    --------
    Dict
        Função valor V estimada
    
    Exemplo:
    --------
    >>> V = first_visit_mc_prediction(gw, policy, n_episodes=1000)
    """
    V = {state: 0.0 for state in gridworld.states}
    gamma = gridworld.gamma
    
    for episode in range(n_episodes):
        # Gera episódio completo
        episode_data = []
        
        if initial_state is not None:
            state = initial_state
        else:
            non_terminal = [s for s in gridworld.states
                           if not gridworld.is_terminal(s)]
            state = non_terminal[np.random.randint(len(non_terminal))]
        
        while not gridworld.is_terminal(state):
            action = policy[state]
            next_state, reward = gridworld.sample_transition(state, action)
            episode_data.append((state, reward))
            state = next_state
        
        # Calcula retornos e atualiza
        G = 0.0
        visited_states = set()
        
        for t in range(len(episode_data) - 1, -1, -1):
            state, reward = episode_data[t]
            G = reward + gamma * G
            
            if state not in visited_states:
                visited_states.add(state)
                V[state] = V[state] + alpha * (G - V[state])
        
        if verbose and (episode + 1) % 100 == 0:
            avg_value = np.mean([V[s] for s in gridworld.states
                                if not gridworld.is_terminal(s)])
            print(f"Episódio {episode + 1}/{n_episodes} - V médio: {avg_value:.4f}")
    
    return V


# ============================================================================
# ALGORITMOS DE CONTROLE (BUSCA DE POLÍTICA ÓTIMA)
# ============================================================================

def sarsa(gridworld: GridWorld, n_episodes: int = 1000,
          alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1,
          q_init: float = 0.0, initial_state: Tuple[int, int] = None,
          verbose: bool = False) -> Tuple[np.ndarray, List[float]]:
    """
    Algoritmo SARSA para controle on-policy.
    
    Aprende Q*(s,a) seguindo política ε-greedy.
    Atualiza Q usando a ação que será realmente tomada.
    
    Fórmula:
    --------
    Q(St, At) ← Q(St, At) + α[Rt+1 + γQ(St+1, At+1) - Q(St, At)]
    
    Parâmetros:
    -----------
    gridworld : GridWorld
        Ambiente
    n_episodes : int, default=1000
        Número de episódios
    alpha : float, default=0.1
        Taxa de aprendizado
        - Valores típicos: 0.01 a 0.5
    gamma : float, default=0.9
        Fator de desconto
        - Valores típicos: 0.9 a 0.99
    epsilon : float, default=0.1
        Probabilidade de exploração
        - Valores típicos: 0.01 a 0.3
    q_init : float, default=0.0
        Valor inicial para Q(s,a)
    initial_state : Tuple[int, int], opcional
        Estado inicial (se None, escolhe aleatoriamente)
    verbose : bool, default=False
        Se True, imprime progresso
    
    Retorna:
    --------
    Tuple[np.ndarray, List[float]]
        - Q: Tabela Q de shape (n_states, n_actions)
        - episode_rewards: Lista com retorno total de cada episódio
    
    Exemplo:
    --------
    >>> Q, rewards = sarsa(gw, n_episodes=1000, alpha=0.1, epsilon=0.1)
    >>> policy = get_greedy_policy(Q, gw)
    """
    n_states = gridworld.rows * gridworld.cols
    n_actions = len(gridworld.actions)
    
    Q = np.full((n_states, n_actions), q_init)
    episode_rewards = []
    
    for episode in range(n_episodes):
        if initial_state is not None:
            state = initial_state
        else:
            non_terminal = [s for s in gridworld.states
                           if not gridworld.is_terminal(s)]
            state = non_terminal[np.random.randint(len(non_terminal))]
        
        action_idx = epsilon_greedy_action(Q, state, gridworld, epsilon)
        total_reward = 0
        
        while not gridworld.is_terminal(state):
            action = gridworld.actions[action_idx]
            next_state, reward = gridworld.sample_transition(state, action)
            total_reward += reward
            
            next_action_idx = epsilon_greedy_action(Q, next_state, gridworld, epsilon)
            
            state_idx = state_to_idx(state, gridworld.cols)
            next_state_idx = state_to_idx(next_state, gridworld.cols)
            
            # Atualização SARSA
            if not gridworld.is_terminal(next_state):
                td_target = reward + gamma * Q[next_state_idx, next_action_idx]
            else:
                td_target = reward
            
            td_error = td_target - Q[state_idx, action_idx]
            Q[state_idx, action_idx] += alpha * td_error
            
            state = next_state
            action_idx = next_action_idx
        
        episode_rewards.append(total_reward)
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episódio {episode + 1}/{n_episodes} - Reward médio: {avg_reward:.2f}")
    
    return Q, episode_rewards


def q_learning(gridworld: GridWorld, n_episodes: int = 1000,
               alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1,
               q_init: float = 0.0, initial_state: Tuple[int, int] = None,
               verbose: bool = False) -> Tuple[np.ndarray, List[float]]:
    """
    Algoritmo Q-Learning para controle off-policy.
    
    Aprende Q*(s,a) - política ótima.
    Atualiza Q usando o máximo Q-value do próximo estado.
    
    Fórmula:
    --------
    Q(St, At) ← Q(St, At) + α[Rt+1 + γ max_a Q(St+1, a) - Q(St, At)]
    
    Parâmetros: mesmos que SARSA
    
    Retorna:
    --------
    Tuple[np.ndarray, List[float]]
        - Q: Tabela Q ótima
        - episode_rewards: Lista de recompensas por episódio
    
    Exemplo:
    --------
    >>> Q, rewards = q_learning(gw, n_episodes=1000, alpha=0.1, epsilon=0.1)
    """
    n_states = gridworld.rows * gridworld.cols
    n_actions = len(gridworld.actions)
    
    Q = np.full((n_states, n_actions), q_init)
    episode_rewards = []
    
    for episode in range(n_episodes):
        if initial_state is not None:
            state = initial_state
        else:
            non_terminal = [s for s in gridworld.states
                           if not gridworld.is_terminal(s)]
            state = non_terminal[np.random.randint(len(non_terminal))]
        
        total_reward = 0
        
        while not gridworld.is_terminal(state):
            action_idx = epsilon_greedy_action(Q, state, gridworld, epsilon)
            action = gridworld.actions[action_idx]
            
            next_state, reward = gridworld.sample_transition(state, action)
            total_reward += reward
            
            state_idx = state_to_idx(state, gridworld.cols)
            next_state_idx = state_to_idx(next_state, gridworld.cols)
            
            # Atualização Q-Learning (usa MAX)
            if not gridworld.is_terminal(next_state):
                max_next_q = np.max(Q[next_state_idx])
                td_target = reward + gamma * max_next_q
            else:
                td_target = reward
            
            td_error = td_target - Q[state_idx, action_idx]
            Q[state_idx, action_idx] += alpha * td_error
            
            state = next_state
        
        episode_rewards.append(total_reward)
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episódio {episode + 1}/{n_episodes} - Reward médio: {avg_reward:.2f}")
    
    return Q, episode_rewards


def expected_sarsa(gridworld: GridWorld, n_episodes: int = 1000,
                   alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1,
                   q_init: float = 0.0, initial_state: Tuple[int, int] = None,
                   verbose: bool = False) -> Tuple[np.ndarray, List[float]]:
    """
    Algoritmo Expected SARSA.
    
    Usa expectativa de Q sob política ε-greedy.
    Reduz variância em relação a SARSA.
    
    Fórmula:
    --------
    Q(St, At) ← Q(St, At) + α[Rt+1 + γ E[Q(St+1, a)] - Q(St, At)]
    onde E[Q(St+1, a)] = Σ_a π(a|St+1) Q(St+1, a)
    
    Parâmetros: mesmos que SARSA
    
    Retorna:
    --------
    Tuple[np.ndarray, List[float]]
        - Q: Tabela Q aprendida
        - episode_rewards: Lista de recompensas
    
    Exemplo:
    --------
    >>> Q, rewards = expected_sarsa(gw, n_episodes=1000, alpha=0.1)
    """
    n_states = gridworld.rows * gridworld.cols
    n_actions = len(gridworld.actions)
    
    Q = np.full((n_states, n_actions), q_init)
    episode_rewards = []
    
    def expected_q(state_idx):
        """Calcula E[Q(s,a)] sob política ε-greedy."""
        q_values = Q[state_idx]
        best_action = np.argmax(q_values)
        
        probs = np.ones(n_actions) * epsilon / n_actions
        probs[best_action] += (1.0 - epsilon)
        
        return np.sum(probs * q_values)
    
    for episode in range(n_episodes):
        if initial_state is not None:
            state = initial_state
        else:
            non_terminal = [s for s in gridworld.states
                           if not gridworld.is_terminal(s)]
            state = non_terminal[np.random.randint(len(non_terminal))]
        
        total_reward = 0
        
        while not gridworld.is_terminal(state):
            action_idx = epsilon_greedy_action(Q, state, gridworld, epsilon)
            action = gridworld.actions[action_idx]
            
            next_state, reward = gridworld.sample_transition(state, action)
            total_reward += reward
            
            state_idx = state_to_idx(state, gridworld.cols)
            next_state_idx = state_to_idx(next_state, gridworld.cols)
            
            # Atualização Expected SARSA
            if not gridworld.is_terminal(next_state):
                expected_next_q = expected_q(next_state_idx)
                td_target = reward + gamma * expected_next_q
            else:
                td_target = reward
            
            td_error = td_target - Q[state_idx, action_idx]
            Q[state_idx, action_idx] += alpha * td_error
            
            state = next_state
        
        episode_rewards.append(total_reward)
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episódio {episode + 1}/{n_episodes} - Reward médio: {avg_reward:.2f}")
    
    return Q, episode_rewards


def first_visit_mc_control(gridworld: GridWorld, n_episodes: int = 1000,
                           alpha: float = 0.1, epsilon: float = 0.1,
                           q_init: float = 0.0, initial_state: Tuple[int, int] = None,
                           verbose: bool = False) -> Tuple[np.ndarray, List[float]]:
    """
    Monte Carlo First-Visit Control com ε-greedy.
    
    Aprende Q(s,a) usando episódios completos.
    Atualiza apenas na primeira visita a cada par (s,a).
    
    Parâmetros:
    -----------
    gridworld : GridWorld
        Ambiente
    n_episodes : int, default=1000
        Número de episódios
    alpha : float, default=0.1
        Taxa de aprendizado
    epsilon : float, default=0.1
        Probabilidade de exploração
    q_init : float, default=0.0
        Valor inicial para Q(s,a)
    initial_state : Tuple[int, int], opcional
        Estado inicial
    verbose : bool, default=False
        Se True, imprime progresso
    
    Retorna:
    --------
    Tuple[np.ndarray, List[float]]
        - Q: Tabela Q aprendida
        - episode_rewards: Lista de recompensas
    
    Exemplo:
    --------
    >>> Q, rewards = first_visit_mc_control(gw, n_episodes=1000)
    """
    n_states = gridworld.rows * gridworld.cols
    n_actions = len(gridworld.actions)
    
    Q = np.full((n_states, n_actions), q_init)
    episode_rewards = []
    gamma = gridworld.gamma
    
    for episode in range(n_episodes):
        episode_data = []
        
        if initial_state is not None:
            state = initial_state
        else:
            non_terminal = [s for s in gridworld.states
                           if not gridworld.is_terminal(s)]
            state = non_terminal[np.random.randint(len(non_terminal))]
        
        total_reward = 0
        
        while not gridworld.is_terminal(state):
            action_idx = epsilon_greedy_action(Q, state, gridworld, epsilon)
            action = gridworld.actions[action_idx]
            
            next_state, reward = gridworld.sample_transition(state, action)
            total_reward += reward
            
            episode_data.append((state, action_idx, reward))
            state = next_state
        
        episode_rewards.append(total_reward)
        
        # Atualiza Q com retornos
        G = 0.0
        visited_pairs = set()
        
        for t in range(len(episode_data) - 1, -1, -1):
            state, action_idx, reward = episode_data[t]
            G = reward + gamma * G
            
            state_action = (state, action_idx)
            
            if state_action not in visited_pairs:
                visited_pairs.add(state_action)
                state_idx = state_to_idx(state, gridworld.cols)
                Q[state_idx, action_idx] += alpha * (G - Q[state_idx, action_idx])
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episódio {episode + 1}/{n_episodes} - Reward médio: {avg_reward:.2f}")
    
    return Q, episode_rewards


def mc_exploring_starts(gridworld: GridWorld, n_episodes: int = 1000,
                       alpha: float = 0.1, q_init: float = 0.0,
                       max_steps: int = 1000,
                       verbose: bool = False) -> Tuple[np.ndarray, List[float]]:
    """
    Monte Carlo com Exploring Starts.
    
    Garante exploração começando de pares (s,a) aleatórios.
    Depois do primeiro passo, segue política gulosa.
    
    Parâmetros:
    -----------
    gridworld : GridWorld
        Ambiente
    n_episodes : int, default=1000
        Número de episódios
    alpha : float, default=0.1
        Taxa de aprendizado
    q_init : float, default=0.0
        Valor inicial para Q(s,a)
        Dica: Use valores pequenos aleatórios (ex: 0.01) para quebrar empates
    max_steps : int, default=1000
        Máximo de passos por episódio (evita loops infinitos)
    verbose : bool, default=False
        Se True, imprime progresso
    
    Retorna:
    --------
    Tuple[np.ndarray, List[float]]
        - Q: Tabela Q aprendida
        - episode_rewards: Lista de recompensas
    
    Exemplo:
    --------
    >>> Q, rewards = mc_exploring_starts(gw, n_episodes=1000, max_steps=500)
    
    Nota:
    -----
    Se os episódios estão demorando muito, considere:
    1. Usar q_init pequeno e aleatório: q_init=0.01
    2. Diminuir max_steps: max_steps=100
    3. Verificar se o ambiente tem solução possível
    """
    n_states = gridworld.rows * gridworld.cols
    n_actions = len(gridworld.actions)
    
    # Inicialização: valores pequenos aleatórios para quebrar empates
    if q_init == 0.0:
        Q = np.random.randn(n_states, n_actions) * 0.01
    else:
        Q = np.full((n_states, n_actions), q_init)
    
    episode_rewards = []
    gamma = gridworld.gamma
    
    for episode in range(n_episodes):
        # EXPLORING START
        non_terminal = [s for s in gridworld.states
                       if not gridworld.is_terminal(s)]
        state = non_terminal[np.random.randint(len(non_terminal))]
        first_action_idx = np.random.randint(n_actions)
        
        episode_data = []
        total_reward = 0
        
        # Primeiro passo: ação aleatória
        action = gridworld.actions[first_action_idx]
        next_state, reward = gridworld.sample_transition(state, action)
        total_reward += reward
        episode_data.append((state, first_action_idx, reward))
        state = next_state
        
        # Resto: política gulosa COM LIMITE DE PASSOS
        steps = 0
        while not gridworld.is_terminal(state) and steps < max_steps:
            state_idx = state_to_idx(state, gridworld.cols)
            
            # Política gulosa com desempate aleatório
            q_values = Q[state_idx]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            action_idx = np.random.choice(best_actions)
            
            action = gridworld.actions[action_idx]
            
            next_state, reward = gridworld.sample_transition(state, action)
            total_reward += reward
            episode_data.append((state, action_idx, reward))
            state = next_state
            steps += 1
        
        # Se atingiu max_steps, aplica penalidade
        if steps >= max_steps and not gridworld.is_terminal(state):
            if verbose and episode < 10:
                print(f"⚠️  Episódio {episode+1}: Atingiu max_steps={max_steps}")
        
        episode_rewards.append(total_reward)
        
        # Atualiza Q
        G = 0.0
        visited_pairs = set()
        
        for t in range(len(episode_data) - 1, -1, -1):
            state, action_idx, reward = episode_data[t]
            G = reward + gamma * G
            
            state_action = (state, action_idx)
            
            if state_action not in visited_pairs:
                visited_pairs.add(state_action)
                state_idx = state_to_idx(state, gridworld.cols)
                Q[state_idx, action_idx] += alpha * (G - Q[state_idx, action_idx])
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episódio {episode + 1}/{n_episodes} - Reward médio: {avg_reward:.2f}")
    
    return Q, episode_rewards
