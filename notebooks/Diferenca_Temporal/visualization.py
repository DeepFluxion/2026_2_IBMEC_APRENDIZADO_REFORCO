"""
visualization.py
================

Funções de visualização para ambientes GridWorld e resultados de RL.

Funções de Visualização de Ambiente:
-------------------------------------
- visualize_gridworld(): Visualiza grid com valores e/ou política
- visualize_q_values(): Visualiza valores máximos por estado
- visualize_q_table_detailed(): Visualiza Q-values de todas as ações

Funções de Análise:
-------------------
- plot_learning_curves(): Plota curvas de aprendizado
- plot_value_evolution(): Mostra evolução de valores ao longo do tempo
- compare_algorithms(): Compara múltiplos algoritmos
- print_q_table(): Imprime tabela Q formatada

Funções de Heatmap:
-------------------
- plot_value_heatmap(): Heatmap de valores de estado
- plot_q_value_heatmap(): Heatmap de Q-values

Autor: Material Educacional RL
Data: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Tuple
from environment import GridWorld


# ============================================================================
# VISUALIZAÇÃO DO GRIDWORLD
# ============================================================================

def visualize_gridworld(gridworld: GridWorld,
                       values: Dict = None,
                       policy: Dict = None,
                       title: str = "GridWorld",
                       figsize: Tuple[int, int] = None):
    """
    Visualiza o GridWorld mostrando valores e/ou política.
    
    Parâmetros:
    -----------
    gridworld : GridWorld
        Instância do ambiente
    values : Dict, opcional
        Dicionário {estado: valor} para exibir valores nos estados
    policy : Dict, opcional
        Dicionário {estado: ação} para exibir política
    title : str, default="GridWorld"
        Título do gráfico
    figsize : Tuple[int, int], opcional
        Tamanho da figura (width, height)
    
    Exemplo:
    --------
    >>> from environment import create_classic_gridworld
    >>> gw = create_classic_gridworld()
    >>> V = {(0,0): 0.5, (0,1): 0.7, (0,2): 0.9}
    >>> visualize_gridworld(gw, values=V, title="Valores V(s)")
    """
    if figsize is None:
        figsize = (gridworld.cols * 2, gridworld.rows * 2)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.set_xlim(-0.5, gridworld.cols - 0.5)
    ax.set_ylim(gridworld.rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(gridworld.cols))
    ax.set_yticks(range(gridworld.rows))
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    action_symbols = {'N': '↑', 'S': '↓', 'L': '→', 'O': '←'}
    
    for row in range(gridworld.rows):
        for col in range(gridworld.cols):
            state = (row, col)
            
            # Determina cor
            if state in gridworld.walls:
                color, alpha = 'gray', 0.8
            elif state in gridworld.terminal_states:
                reward = gridworld.terminal_states[state]
                color = 'lightgreen' if reward > 0 else 'lightcoral'
                alpha = 0.6
            else:
                color, alpha = 'white', 0.3
            
            # Desenha célula
            rect = FancyBboxPatch(
                (col - 0.45, row - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor='black',
                alpha=alpha, linewidth=2
            )
            ax.add_patch(rect)
            
            # Conteúdo
            if state in gridworld.walls:
                ax.text(col, row, '■', ha='center', va='center',
                       fontsize=20, fontweight='bold')
            elif state in gridworld.terminal_states:
                reward = gridworld.terminal_states[state]
                ax.text(col, row, f'{reward:+.0f}', ha='center', va='center',
                       fontsize=16, fontweight='bold')
            else:
                if values is not None and state in values:
                    ax.text(col, row - 0.25, f'{values[state]:.3f}',
                           ha='center', va='center', fontsize=10)
                if policy is not None and state in policy:
                    action = policy[state]
                    if action in action_symbols:
                        ax.text(col, row + 0.1, action_symbols[action],
                               ha='center', va='center', fontsize=20,
                               fontweight='bold', color='blue')
    
    plt.tight_layout()
    plt.show()


def visualize_q_values(Q: np.ndarray, gridworld: GridWorld,
                      title: str = "Q-Values por Estado",
                      figsize: Tuple[int, int] = None):
    """
    Visualiza os valores máximos de Q para cada estado.
    
    Mostra max_a Q(s,a) para cada estado.
    
    Parâmetros:
    -----------
    Q : np.ndarray
        Tabela Q de shape (n_states, n_actions)
    gridworld : GridWorld
        Ambiente
    title : str
        Título do gráfico
    figsize : Tuple[int, int], opcional
        Tamanho da figura
    
    Exemplo:
    --------
    >>> from algorithms import q_learning
    >>> Q, _ = q_learning(gw, n_episodes=1000)
    >>> visualize_q_values(Q, gw, title="Q-Learning - Valores")
    """
    values = {}
    for state in gridworld.states:
        if not gridworld.is_terminal(state):
            state_idx = state[0] * gridworld.cols + state[1]
            values[state] = np.max(Q[state_idx])
    
    visualize_gridworld(gridworld, values=values, title=title, figsize=figsize)


def visualize_q_table_detailed(Q: np.ndarray, gridworld: GridWorld,
                               title: str = "Tabela Q Detalhada",
                               figsize: Tuple[int, int] = None):
    """
    Visualiza todos os Q-values para cada estado e ação.
    
    Mostra Q(s,a) para todas as 4 ações em cada célula.
    A melhor ação é destacada em verde e negrito.
    
    Parâmetros:
    -----------
    Q : np.ndarray
        Tabela Q
    gridworld : GridWorld
        Ambiente
    title : str
        Título do gráfico
    figsize : Tuple[int, int], opcional
        Tamanho da figura
    
    Exemplo:
    --------
    >>> visualize_q_table_detailed(Q, gw, title="Q-Values Detalhados")
    """
    if figsize is None:
        figsize = (gridworld.cols * 3, gridworld.rows * 3)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.set_xlim(-0.5, gridworld.cols - 0.5)
    ax.set_ylim(gridworld.rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(gridworld.cols))
    ax.set_yticks(range(gridworld.rows))
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    action_labels = {'N': '↑', 'S': '↓', 'L': '→', 'O': '←'}
    
    for row in range(gridworld.rows):
        for col in range(gridworld.cols):
            state = (row, col)
            
            # Cor de fundo
            if state in gridworld.walls:
                color, alpha = 'gray', 0.8
            elif state in gridworld.terminal_states:
                reward = gridworld.terminal_states[state]
                color = 'lightgreen' if reward > 0 else 'lightcoral'
                alpha = 0.6
            else:
                color, alpha = 'white', 0.3
            
            rect = FancyBboxPatch(
                (col - 0.45, row - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor='black',
                alpha=alpha, linewidth=2
            )
            ax.add_patch(rect)
            
            if state in gridworld.walls:
                ax.text(col, row, '■', ha='center', va='center',
                       fontsize=20, fontweight='bold')
            elif state in gridworld.terminal_states:
                reward = gridworld.terminal_states[state]
                ax.text(col, row, f'{reward:+.0f}', ha='center', va='center',
                       fontsize=16, fontweight='bold')
            else:
                state_idx = row * gridworld.cols + col
                
                positions = {
                    'N': (col, row - 0.3),
                    'S': (col, row + 0.3),
                    'L': (col + 0.25, row),
                    'O': (col - 0.25, row)
                }
                
                best_action = np.argmax(Q[state_idx])
                
                for action_idx, action in enumerate(gridworld.actions):
                    q_val = Q[state_idx, action_idx]
                    x, y = positions[action]
                    
                    color = 'green' if action_idx == best_action else 'black'
                    weight = 'bold' if action_idx == best_action else 'normal'
                    
                    ax.text(x, y, f'{action_labels[action]}{q_val:.2f}',
                           ha='center', va='center', fontsize=8,
                           color=color, weight=weight)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# CURVAS DE APRENDIZADO
# ============================================================================

def plot_learning_curves(rewards_dict: Dict[str, List[float]],
                         window: int = 100,
                         title: str = "Curvas de Aprendizado",
                         figsize: Tuple[int, int] = (12, 6)):
    """
    Plota curvas de aprendizado para múltiplos algoritmos.
    
    Mostra a média móvel das recompensas ao longo dos episódios.
    
    Parâmetros:
    -----------
    rewards_dict : Dict[str, List[float]]
        Dicionário {nome_algoritmo: lista_recompensas}
    window : int, default=100
        Janela para média móvel
    title : str
        Título do gráfico
    figsize : Tuple[int, int]
        Tamanho da figura
    
    Exemplo:
    --------
    >>> from algorithms import sarsa, q_learning
    >>> Q_s, rewards_s = sarsa(gw, n_episodes=1000)
    >>> Q_q, rewards_q = q_learning(gw, n_episodes=1000)
    >>> plot_learning_curves({
    ...     'SARSA': rewards_s,
    ...     'Q-Learning': rewards_q
    ... })
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Gráfico 1: Recompensas brutas
    for name, rewards in rewards_dict.items():
        ax1.plot(rewards, alpha=0.3, label=f'{name} (bruto)')
    
    ax1.set_xlabel('Episódio')
    ax1.set_ylabel('Recompensa Total')
    ax1.set_title('Recompensas por Episódio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Média móvel
    for name, rewards in rewards_dict.items():
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax2.plot(smoothed, label=name, linewidth=2)
    
    ax2.set_xlabel('Episódio')
    ax2.set_ylabel(f'Recompensa Média (janela={window})')
    ax2.set_title('Curvas de Aprendizado Suavizadas')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_value_evolution(value_history: List[Dict],
                         states: List[Tuple[int, int]],
                         title: str = "Evolução dos Valores",
                         figsize: Tuple[int, int] = (12, 6)):
    """
    Plota evolução dos valores de estados específicos ao longo do tempo.
    
    Parâmetros:
    -----------
    value_history : List[Dict]
        Lista de dicionários V em diferentes momentos
        Cada elemento: {estado: valor}
    states : List[Tuple[int, int]]
        Estados a monitorar
    title : str
        Título do gráfico
    figsize : Tuple[int, int]
        Tamanho da figura
    
    Exemplo:
    --------
    >>> # Durante treinamento, salvar V periodicamente
    >>> value_history = []
    >>> for ep in range(0, n_episodes, 100):
    ...     # treinar...
    ...     value_history.append(V.copy())
    >>> plot_value_evolution(value_history, [(0,0), (1,0), (2,0)])
    """
    plt.figure(figsize=figsize)
    
    for state in states:
        values = [V[state] for V in value_history if state in V]
        plt.plot(values, marker='o', label=f'Estado {state}', linewidth=2)
    
    plt.xlabel('Checkpoint (a cada N episódios)')
    plt.ylabel('Valor V(s)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# HEATMAPS
# ============================================================================

def plot_value_heatmap(values: Dict, gridworld: GridWorld,
                      title: str = "Heatmap de Valores",
                      figsize: Tuple[int, int] = None,
                      cmap: str = 'RdYlGn'):
    """
    Cria heatmap dos valores de estado.
    
    Parâmetros:
    -----------
    values : Dict
        Dicionário {estado: valor}
    gridworld : GridWorld
        Ambiente
    title : str
        Título do gráfico
    figsize : Tuple[int, int], opcional
        Tamanho da figura
    cmap : str, default='RdYlGn'
        Colormap do matplotlib
    
    Exemplo:
    --------
    >>> from algorithms import td_zero_prediction
    >>> policy = {s: 'N' for s in gw.states if not gw.is_terminal(s)}
    >>> V = td_zero_prediction(gw, policy, n_episodes=1000)
    >>> plot_value_heatmap(V, gw, title="TD(0) - Valores")
    """
    if figsize is None:
        figsize = (gridworld.cols * 1.5, gridworld.rows * 1.5)
    
    # Cria matriz de valores
    value_matrix = np.full((gridworld.rows, gridworld.cols), np.nan)
    
    for state, value in values.items():
        if state not in gridworld.terminal_states and state not in gridworld.walls:
            value_matrix[state[0], state[1]] = value
    
    plt.figure(figsize=figsize)
    im = plt.imshow(value_matrix, cmap=cmap, aspect='auto')
    
    # Adiciona valores nas células
    for row in range(gridworld.rows):
        for col in range(gridworld.cols):
            state = (row, col)
            if state in gridworld.walls:
                plt.text(col, row, '■', ha='center', va='center',
                        fontsize=20, fontweight='bold')
            elif state in gridworld.terminal_states:
                reward = gridworld.terminal_states[state]
                plt.text(col, row, f'{reward:+.0f}', ha='center', va='center',
                        fontsize=14, fontweight='bold', color='white')
            elif not np.isnan(value_matrix[row, col]):
                plt.text(col, row, f'{value_matrix[row, col]:.2f}',
                        ha='center', va='center', fontsize=10)
    
    plt.colorbar(im, label='Valor')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Coluna')
    plt.ylabel('Linha')
    plt.tight_layout()
    plt.show()


def plot_q_value_heatmap(Q: np.ndarray, gridworld: GridWorld,
                        action: str = None,
                        title: str = "Heatmap de Q-Values",
                        figsize: Tuple[int, int] = None,
                        cmap: str = 'viridis'):
    """
    Cria heatmap dos Q-values.
    
    Parâmetros:
    -----------
    Q : np.ndarray
        Tabela Q
    gridworld : GridWorld
        Ambiente
    action : str, opcional
        Se fornecido, mostra Q(s, action). Se None, mostra max_a Q(s,a)
    title : str
        Título do gráfico
    figsize : Tuple[int, int], opcional
        Tamanho da figura
    cmap : str, default='viridis'
        Colormap
    
    Exemplo:
    --------
    >>> # Heatmap dos valores máximos
    >>> plot_q_value_heatmap(Q, gw, title="Max Q-Values")
    >>> 
    >>> # Heatmap para ação específica
    >>> plot_q_value_heatmap(Q, gw, action='N', title="Q(s, Norte)")
    """
    if figsize is None:
        figsize = (gridworld.cols * 1.5, gridworld.rows * 1.5)
    
    value_matrix = np.full((gridworld.rows, gridworld.cols), np.nan)
    
    for state in gridworld.states:
        if state not in gridworld.terminal_states and state not in gridworld.walls:
            state_idx = state[0] * gridworld.cols + state[1]
            
            if action is not None:
                action_idx = gridworld.actions.index(action)
                value_matrix[state[0], state[1]] = Q[state_idx, action_idx]
            else:
                value_matrix[state[0], state[1]] = np.max(Q[state_idx])
    
    plt.figure(figsize=figsize)
    im = plt.imshow(value_matrix, cmap=cmap, aspect='auto')
    
    for row in range(gridworld.rows):
        for col in range(gridworld.cols):
            state = (row, col)
            if state in gridworld.walls:
                plt.text(col, row, '■', ha='center', va='center',
                        fontsize=20, fontweight='bold')
            elif state in gridworld.terminal_states:
                reward = gridworld.terminal_states[state]
                plt.text(col, row, f'{reward:+.0f}', ha='center', va='center',
                        fontsize=14, fontweight='bold', color='white')
            elif not np.isnan(value_matrix[row, col]):
                plt.text(col, row, f'{value_matrix[row, col]:.2f}',
                        ha='center', va='center', fontsize=9)
    
    plt.colorbar(im, label='Q-Value')
    
    if action:
        full_title = f"{title} - Ação: {action}"
    else:
        full_title = f"{title} - Max Q"
    
    plt.title(full_title, fontsize=14, fontweight='bold')
    plt.xlabel('Coluna')
    plt.ylabel('Linha')
    plt.tight_layout()
    plt.show()


# ============================================================================
# COMPARAÇÃO E ANÁLISE
# ============================================================================

def compare_algorithms(Q_dict: Dict[str, np.ndarray], gridworld: GridWorld):
    """
    Compara valores aprendidos por diferentes algoritmos.
    
    Imprime tabela comparativa e estatísticas.
    
    Parâmetros:
    -----------
    Q_dict : Dict[str, np.ndarray]
        Dicionário {nome_algoritmo: tabela_Q}
    gridworld : GridWorld
        Ambiente
    
    Exemplo:
    --------
    >>> compare_algorithms({
    ...     'SARSA': Q_sarsa,
    ...     'Q-Learning': Q_qlearning,
    ...     'Expected SARSA': Q_expected
    ... }, gw)
    """
    print("="*80)
    print("COMPARAÇÃO DE VALORES APRENDIDOS")
    print("="*80)
    print(f"\n{'Estado':<15}", end='')
    
    for algo_name in Q_dict.keys():
        print(f"{algo_name:<20}", end='')
    print()
    print("-"*80)
    
    for state in gridworld.states:
        if not gridworld.is_terminal(state) and state not in gridworld.walls:
            state_idx = state[0] * gridworld.cols + state[1]
            print(f"{str(state):<15}", end='')
            
            for Q in Q_dict.values():
                max_q = np.max(Q[state_idx])
                print(f"{max_q:<20.4f}", end='')
            print()
    
    print("="*80)
    print("\nESTATÍSTICAS RESUMIDAS:")
    print("-"*80)
    
    for algo_name, Q in Q_dict.items():
        values = []
        for state in gridworld.states:
            if not gridworld.is_terminal(state) and state not in gridworld.walls:
                state_idx = state[0] * gridworld.cols + state[1]
                values.append(np.max(Q[state_idx]))
        
        print(f"\n{algo_name}:")
        print(f"  Valor médio:   {np.mean(values):.4f}")
        print(f"  Valor máximo:  {np.max(values):.4f}")
        print(f"  Valor mínimo:  {np.min(values):.4f}")
        print(f"  Desvio padrão: {np.std(values):.4f}")


def print_q_table(Q: np.ndarray, gridworld: GridWorld,
                 title: str = "Tabela Q"):
    """
    Imprime a tabela Q de forma legível no console.
    
    Parâmetros:
    -----------
    Q : np.ndarray
        Tabela Q
    gridworld : GridWorld
        Ambiente
    title : str
        Título da tabela
    
    Exemplo:
    --------
    >>> print_q_table(Q, gw, title="SARSA - Tabela Q")
    """
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)
    
    for state in gridworld.states:
        if gridworld.is_terminal(state):
            print(f"\nEstado {state}: TERMINAL")
            continue
        
        if state in gridworld.walls:
            print(f"\nEstado {state}: PAREDE")
            continue
        
        state_idx = state[0] * gridworld.cols + state[1]
        print(f"\nEstado {state}:")
        
        for action_idx, action in enumerate(gridworld.actions):
            q_val = Q[state_idx, action_idx]
            symbol = "  " if action_idx != np.argmax(Q[state_idx]) else "→"
            print(f"  {symbol} {action}: {q_val:.4f}")
        
        best_action = gridworld.actions[np.argmax(Q[state_idx])]
        print(f"  Melhor ação: {best_action}")
    
    print("="*80)
