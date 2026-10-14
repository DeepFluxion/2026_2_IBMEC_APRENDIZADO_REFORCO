"""
environment.py
==============

Módulo para criação e manipulação de ambientes GridWorld para Aprendizado por Reforço.

Classes:
--------
- GridWorld: Ambiente de grid com transições estocásticas

Funções:
--------
- create_classic_gridworld(): Cria o GridWorld 4x3 clássico do Russell & Norvig
- create_custom_gridworld(): Template para criar ambientes personalizados

Autor: Material Educacional RL
Data: 2025
"""

import numpy as np
from typing import Tuple, Dict, List, Set


class GridWorld:
    """
    Ambiente GridWorld para experimentos de Aprendizado por Reforço.
    
    O agente pode se mover em 4 direções (N, S, L, O) com ruído estocástico.
    
    Atributos:
    ----------
    rows : int
        Número de linhas do grid
    cols : int
        Número de colunas do grid
    gamma : float
        Fator de desconto (0 < γ ≤ 1)
    noise : float
        Probabilidade total de ruído no movimento
    actions : list
        Lista de ações disponíveis ['N', 'S', 'L', 'O']
    states : list
        Lista de estados válidos [(row, col), ...]
    walls : set
        Conjunto de estados que são paredes
    terminal_states : dict
        Dicionário {estado: recompensa} para estados terminais
    living_reward : float
        Recompensa padrão por cada passo (custo de viver)
    
    Exemplo:
    --------
    >>> gw = GridWorld(3, 4, gamma=0.9, noise=0.2)
    >>> gw.set_wall(1, 1)
    >>> gw.set_terminal(0, 3, 1.0)
    >>> gw.set_terminal(1, 3, -1.0)
    >>> state, reward = gw.sample_transition((2, 0), 'N')
    """
    
    def __init__(self, rows: int, cols: int, gamma: float = 0.9, noise: float = 0.2):
        """
        Inicializa o GridWorld.
        
        Parâmetros:
        -----------
        rows : int
            Número de linhas do grid
        cols : int
            Número de colunas do grid
        gamma : float, default=0.9
            Fator de desconto (0 < γ ≤ 1)
            - γ próximo de 0: valoriza recompensas imediatas
            - γ próximo de 1: valoriza recompensas futuras
        noise : float, default=0.2
            Probabilidade total de ruído no movimento
            - 0.0: determinístico
            - 0.2: 20% de chance de movimento lateral
        """
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.noise = noise
        
        # Ações: Norte, Sul, Leste, Oeste
        self.actions = ['N', 'S', 'L', 'O']
        
        # Efeitos das ações nas coordenadas (delta_row, delta_col)
        self.action_effects = {
            'N': (-1, 0),  # Norte: diminui linha (sobe)
            'S': (1, 0),   # Sul: aumenta linha (desce)
            'L': (0, 1),   # Leste: aumenta coluna (direita)
            'O': (0, -1)   # Oeste: diminui coluna (esquerda)
        }
        
        # Ações perpendiculares para cada ação (para modelar ruído)
        self.perpendicular = {
            'N': ['L', 'O'],
            'S': ['L', 'O'],
            'L': ['N', 'S'],
            'O': ['N', 'S']
        }
        
        # Inicializa todos os estados como válidos
        self.states = [(r, c) for r in range(rows) for c in range(cols)]
        
        # Configurações do mundo (devem ser definidas pelo usuário)
        self.walls: Set[Tuple[int, int]] = set()
        self.terminal_states: Dict[Tuple[int, int], float] = {}
        self.living_reward = -0.04
    
    def set_wall(self, row: int, col: int):
        """
        Define uma posição como parede (estado inválido).
        
        Parâmetros:
        -----------
        row : int
            Linha da parede
        col : int
            Coluna da parede
        """
        state = (row, col)
        if state in self.states:
            self.states.remove(state)
        self.walls.add(state)
    
    def set_terminal(self, row: int, col: int, reward: float):
        """
        Define um estado terminal e sua recompensa.
        
        Parâmetros:
        -----------
        row : int
            Linha do estado terminal
        col : int
            Coluna do estado terminal
        reward : float
            Recompensa ao alcançar este estado
        """
        self.terminal_states[(row, col)] = reward
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """
        Verifica se um estado é terminal.
        
        Parâmetros:
        -----------
        state : Tuple[int, int]
            Estado (row, col)
        
        Retorna:
        --------
        bool
            True se o estado é terminal
        """
        return state in self.terminal_states
    
    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """
        Verifica se um estado é válido (não é parede e está dentro do grid).
        
        Parâmetros:
        -----------
        state : Tuple[int, int]
            Estado (row, col)
        
        Retorna:
        --------
        bool
            True se o estado é válido
        """
        row, col = state
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                state not in self.walls)
    
    def move(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """
        Executa um movimento determinístico.
        
        Retorna o novo estado após aplicar a ação.
        Se o movimento for inválido (parede ou fora do grid), retorna o estado original.
        
        Parâmetros:
        -----------
        state : Tuple[int, int]
            Estado atual
        action : str
            Ação a executar ('N', 'S', 'L', 'O')
        
        Retorna:
        --------
        Tuple[int, int]
            Novo estado após o movimento
        """
        if self.is_terminal(state):
            return state
        
        row, col = state
        drow, dcol = self.action_effects[action]
        new_state = (row + drow, col + dcol)
        
        # Se o novo estado for válido, move; senão, fica parado
        if self.is_valid_state(new_state):
            return new_state
        else:
            return state
    
    def get_transition_prob(self, state: Tuple[int, int], action: str,
                           next_state: Tuple[int, int]) -> float:
        """
        Calcula P(next_state | state, action).
        
        Modelo de transição estocástico:
        - Probabilidade (1-noise): move na direção pretendida
        - Probabilidade noise/2: move para cada lado perpendicular
        
        Parâmetros:
        -----------
        state : Tuple[int, int]
            Estado atual
        action : str
            Ação executada
        next_state : Tuple[int, int]
            Próximo estado
        
        Retorna:
        --------
        float
            Probabilidade da transição
        """
        if self.is_terminal(state):
            return 1.0 if next_state == state else 0.0
        
        # Calcula onde o agente iria com cada ação possível
        intended_state = self.move(state, action)
        
        prob = 0.0
        
        # Se a ação pretendida leva ao next_state
        if intended_state == next_state:
            prob += (1.0 - self.noise)
        
        # Adiciona probabilidade das ações perpendiculares
        for perp_action in self.perpendicular[action]:
            if self.move(state, perp_action) == next_state:
                prob += self.noise / 2
        
        return prob
    
    def get_reward(self, state: Tuple[int, int], action: str,
                   next_state: Tuple[int, int]) -> float:
        """
        Retorna R(state, action, next_state).
        
        Parâmetros:
        -----------
        state : Tuple[int, int]
            Estado atual
        action : str
            Ação executada
        next_state : Tuple[int, int]
            Próximo estado
        
        Retorna:
        --------
        float
            Recompensa da transição
            - Se next_state é terminal: retorna recompensa do terminal
            - Caso contrário: retorna living_reward
        """
        if next_state in self.terminal_states:
            return self.terminal_states[next_state]
        else:
            return self.living_reward
    
    def sample_transition(self, state: Tuple[int, int],
                         action: str) -> Tuple[Tuple[int, int], float]:
        """
        Simula uma transição estocástica.
        
        Executa a ação considerando o ruído do ambiente.
        
        Parâmetros:
        -----------
        state : Tuple[int, int]
            Estado atual
        action : str
            Ação a executar
        
        Retorna:
        --------
        Tuple[Tuple[int, int], float]
            (next_state, reward)
        """
        if self.is_terminal(state):
            return state, 0.0
        
        # Escolhe ação real baseada nas probabilidades
        rand = np.random.random()
        
        if rand < (1.0 - self.noise):
            # Ação pretendida
            actual_action = action
        elif rand < (1.0 - self.noise / 2):
            # Primeira ação perpendicular
            actual_action = self.perpendicular[action][0]
        else:
            # Segunda ação perpendicular
            actual_action = self.perpendicular[action][1]
        
        next_state = self.move(state, actual_action)
        reward = self.get_reward(state, action, next_state)
        
        return next_state, reward
    
    def get_all_transitions(self, state: Tuple[int, int],
                           action: str) -> List[Tuple[float, Tuple[int, int], float]]:
        """
        Retorna todas as possíveis transições de (state, action).
        
        Parâmetros:
        -----------
        state : Tuple[int, int]
            Estado atual
        action : str
            Ação executada
        
        Retorna:
        --------
        List[Tuple[float, Tuple[int, int], float]]
            Lista de tuplas (probabilidade, next_state, reward)
        """
        transitions = []
        
        for next_state in self.states:
            prob = self.get_transition_prob(state, action, next_state)
            if prob > 0:
                reward = self.get_reward(state, action, next_state)
                transitions.append((prob, next_state, reward))
        
        return transitions


def create_classic_gridworld() -> GridWorld:
    """
    Cria o GridWorld 4x3 clássico do Russell & Norvig.
    
    Layout:
    -------
    [ ] [ ] [ ] [+1]
    [ ] [■] [ ] [-1]
    [S] [ ] [ ] [ ]
    
    Legenda:
    --------
    S  : Estado inicial típico (2,0)
    ■  : Parede em (1,1)
    +1 : Terminal positivo em (0,3)
    -1 : Terminal negativo em (1,3)
    
    Configuração:
    -------------
    - Tamanho: 3 linhas x 4 colunas
    - Fator de desconto: γ = 0.9
    - Ruído: 20% (0.2)
    - Living reward: -0.04
    
    Retorna:
    --------
    GridWorld
        Ambiente configurado
    
    Exemplo:
    --------
    >>> gw = create_classic_gridworld()
    >>> print(f"Estados: {len(gw.states)}")
    Estados: 11
    """
    gw = GridWorld(3, 4, gamma=0.9, noise=0.2)
    
    # Define parede
    gw.set_wall(1, 1)
    
    # Define estados terminais
    gw.set_terminal(0, 3, 1.0)   # Terminal positivo
    gw.set_terminal(1, 3, -1.0)  # Terminal negativo
    
    # Custo de viver padrão
    gw.living_reward = -0.04
    
    return gw


def create_custom_gridworld(rows: int, cols: int,
                           walls: List[Tuple[int, int]] = None,
                           terminals: Dict[Tuple[int, int], float] = None,
                           gamma: float = 0.9,
                           noise: float = 0.2,
                           living_reward: float = -0.04) -> GridWorld:
    """
    Cria um GridWorld personalizado.
    
    Parâmetros:
    -----------
    rows : int
        Número de linhas
    cols : int
        Número de colunas
    walls : List[Tuple[int, int]], opcional
        Lista de posições de paredes [(row, col), ...]
    terminals : Dict[Tuple[int, int], float], opcional
        Dicionário de estados terminais {(row, col): reward}
    gamma : float, default=0.9
        Fator de desconto
    noise : float, default=0.2
        Probabilidade de ruído
    living_reward : float, default=-0.04
        Recompensa por passo
    
    Retorna:
    --------
    GridWorld
        Ambiente configurado
    
    Exemplo:
    --------
    >>> # Cria um grid 5x5 com paredes e terminais customizados
    >>> gw = create_custom_gridworld(
    ...     rows=5, cols=5,
    ...     walls=[(1, 1), (1, 2), (2, 1)],
    ...     terminals={(0, 4): 10.0, (4, 0): -10.0},
    ...     gamma=0.95,
    ...     noise=0.1
    ... )
    """
    gw = GridWorld(rows, cols, gamma=gamma, noise=noise)
    
    # Define paredes
    if walls is not None:
        for wall in walls:
            gw.set_wall(wall[0], wall[1])
    
    # Define terminais
    if terminals is not None:
        for state, reward in terminals.items():
            gw.set_terminal(state[0], state[1], reward)
    
    gw.living_reward = living_reward
    
    return gw


def create_cliff_world(rows: int = 4, cols: int = 8, 
                       cliff_reward: float = -100.0,
                       goal_reward: float = 0.0,
                       living_reward: float = -1.0,
                       gamma: float = 0.9,
                       noise: float = 0.0) -> GridWorld:
    """
    Cria o ambiente Cliff World (configurável).
    
    Layout padrão (4x8):
    --------------------
    [ ] [ ] [ ] [ ] [ ] [ ] [ ] [G]
    [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
    [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
    [S] [C] [C] [C] [C] [C] [C] [C]
    
    Legenda:
    --------
    S : Start (última linha, primeira coluna)
    C : Cliff (precipício) - última linha, exceto primeira e última coluna
    G : Goal (primeira linha, última coluna)
    
    Parâmetros:
    -----------
    rows : int, default=4
        Número de linhas
    cols : int, default=8
        Número de colunas
    cliff_reward : float, default=-100.0
        Recompensa por cair no precipício (negativa!)
    goal_reward : float, default=0.0
        Recompensa por alcançar o objetivo
    living_reward : float, default=-1.0
        Custo por passo (incentiva caminhos curtos)
    gamma : float, default=0.9
        Fator de desconto
    noise : float, default=0.0
        Probabilidade de ruído (0.0 = determinístico)
    
    Retorna:
    --------
    GridWorld
        Ambiente Cliff World configurado
    
    Exemplos:
    ---------
    >>> # Cliff World padrão (4x8, determinístico)
    >>> gw = create_cliff_world()
    
    >>> # Cliff World maior
    >>> gw = create_cliff_world(rows=6, cols=12)
    
    >>> # Cliff menos punitivo
    >>> gw = create_cliff_world(cliff_reward=-50.0)
    
    >>> # Cliff com ruído (estocástico)
    >>> gw = create_cliff_world(noise=0.1)
    
    >>> # Cliff com objetivo positivo
    >>> gw = create_cliff_world(goal_reward=10.0, living_reward=-0.1)
    """
    gw = GridWorld(rows, cols, gamma=gamma, noise=noise)
    
    # Define o cliff (precipício) na última linha
    # Exceto primeira coluna (start) e última coluna (próximo ao goal)
    cliff_row = rows - 1
    for col in range(1, cols - 1):
        gw.set_terminal(cliff_row, col, cliff_reward)
    
    # Define o goal no canto superior direito
    gw.set_terminal(0, cols - 1, goal_reward)
    
    # Custo de viver
    gw.living_reward = living_reward
    
    return gw

def create_cliff_world_2(rows: int = 4, cols: int = 8, 
                       cliff_reward: float = -100.0,
                       goal_reward: float = 0.0,
                       living_reward: float = -1.0,
                       gamma: float = 0.9,
                       noise: float = 0.0) -> GridWorld:
    """
    Cria o ambiente Cliff World (configurável).
    
    Layout padrão (4x8):
    --------------------
    [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
    [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
    [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
    [S] [C] [C] [C] [C] [C] [C] [G]
    
    Legenda:
    --------
    S : Start (última linha, primeira coluna)
    C : Cliff (precipício) - última linha, exceto primeira e última coluna
    G : Goal (ultima linha, última coluna)
    
    Parâmetros:
    -----------
    rows : int, default=4
        Número de linhas
    cols : int, default=8
        Número de colunas
    cliff_reward : float, default=-100.0
        Recompensa por cair no precipício (negativa!)
    goal_reward : float, default=0.0
        Recompensa por alcançar o objetivo
    living_reward : float, default=-1.0
        Custo por passo (incentiva caminhos curtos)
    gamma : float, default=0.9
        Fator de desconto
    noise : float, default=0.0
        Probabilidade de ruído (0.0 = determinístico)
    
    Retorna:
    --------
    GridWorld
        Ambiente Cliff World configurado
    
    Exemplos:
    ---------
    >>> # Cliff World padrão (4x8, determinístico)
    >>> gw = create_cliff_world_2()
    
    >>> # Cliff World maior
    >>> gw = create_cliff_world_2(rows=6, cols=12)
    
    >>> # Cliff menos punitivo
    >>> gw = create_cliff_world_2(cliff_reward=-50.0)
    
    >>> # Cliff com ruído (estocástico)
    >>> gw = create_cliff_world_2(noise=0.1)
    
    >>> # Cliff com objetivo positivo
    >>> gw = create_cliff_world_2(goal_reward=10.0, living_reward=-0.1)
    """
    gw = GridWorld(rows, cols, gamma=gamma, noise=noise)
    
    # Define o cliff (precipício) na última linha
    # Exceto primeira coluna (start) e última coluna (próximo ao goal)
    cliff_row = rows - 1
    for col in range(1, cols - 1):
        gw.set_terminal(cliff_row, col, cliff_reward)
    
    # Define o goal no canto superior direito
    gw.set_terminal(rows-1, cols - 1, goal_reward)
    
    # Custo de viver
    gw.living_reward = living_reward
    
    return gw

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def print_gridworld_info(gw: GridWorld):
    """
    Imprime informações sobre o GridWorld.
    
    Parâmetros:
    -----------
    gw : GridWorld
        Ambiente
    """
    print("="*70)
    print("INFORMAÇÕES DO GRIDWORLD")
    print("="*70)
    print(f"Dimensões: {gw.rows} linhas x {gw.cols} colunas")
    print(f"Total de estados: {len(gw.states)}")
    print(f"Paredes: {len(gw.walls)}")
    print(f"Estados terminais: {len(gw.terminal_states)}")
    print(f"Fator de desconto (γ): {gw.gamma}")
    print(f"Ruído: {gw.noise}")
    print(f"Living reward: {gw.living_reward}")
    print(f"Ações disponíveis: {gw.actions}")
    
    if gw.walls:
        print(f"\nParedes: {sorted(gw.walls)}")
    
    if gw.terminal_states:
        print("\nEstados terminais:")
        for state, reward in sorted(gw.terminal_states.items()):
            print(f"  {state}: reward = {reward}")
    
    print("="*70)
