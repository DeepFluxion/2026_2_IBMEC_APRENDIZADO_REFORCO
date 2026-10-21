"""
compare_q_values.py
===================

Função para comparação detalhada de Q-values entre algoritmos TD
com seleção de estados específicos.

Autor: Análise Comparativa RL
Data: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
from environment import GridWorld


def compare_q_values_by_states(
    Q_dict: Dict[str, np.ndarray],
    gridworld: GridWorld,
    selected_states: Optional[List[Tuple[int, int]]] = None,
    show_actions: bool = True,
    show_differences: bool = True,
    plot_comparison: bool = True,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Compara Q-values de múltiplos algoritmos para estados específicos.
    
    Parâmetros:
    -----------
    Q_dict : Dict[str, np.ndarray]
        Dicionário {nome_algoritmo: tabela_Q}
        Exemplo: {'SARSA': Q_sarsa, 'Q-Learning': Q_qlearning}
    
    gridworld : GridWorld
        Instância do ambiente GridWorld
    
    selected_states : List[Tuple[int, int]], opcional
        Lista de estados (linha, coluna) para analisar.
        Se None, analisa todos os estados não-terminais e não-parede.
        Exemplo: [(0,0), (1,1), (2,2)]
    
    show_actions : bool, default=True
        Se True, mostra Q-values de todas as ações.
        Se False, mostra apenas o valor máximo Q(s) = max_a Q(s,a)
    
    show_differences : bool, default=True
        Se True, calcula e mostra as diferenças entre algoritmos
    
    plot_comparison : bool, default=True
        Se True, gera gráfico de barras comparativo
    
    figsize : Tuple[int, int], default=(14, 8)
        Tamanho da figura para o gráfico
    
    Retorna:
    --------
    df_comparison : pd.DataFrame
        DataFrame com os valores comparados
    
    Exemplo de Uso:
    ---------------
    >>> # Comparar 3 algoritmos em estados específicos
    >>> Q_dict = {
    ...     'SARSA': Q_sarsa,
    ...     'Q-Learning': Q_qlearning,
    ...     'Expected SARSA': Q_expected
    ... }
    >>> 
    >>> # Selecionar estados de interesse
    >>> estados = [(0,1), (1,1), (2,1), (3,0)]
    >>> 
    >>> # Executar comparação completa
    >>> df = compare_q_values_by_states(
    ...     Q_dict, gw, 
    ...     selected_states=estados,
    ...     show_actions=True,
    ...     show_differences=True,
    ...     plot_comparison=True
    ... )
    >>> 
    >>> # Comparar apenas valores máximos (sem ações individuais)
    >>> df_simple = compare_q_values_by_states(
    ...     Q_dict, gw, 
    ...     selected_states=estados,
    ...     show_actions=False
    ... )
    """
    
    # Se nenhum estado foi selecionado, usa todos os válidos
    if selected_states is None:
        selected_states = [
            state for state in gridworld.states
            if not gridworld.is_terminal(state) and state not in gridworld.walls
        ]
    
    # Valida estados selecionados
    valid_states = []
    for state in selected_states:
        if state in gridworld.walls:
            print(f"⚠️  Estado {state} é uma PAREDE - ignorado")
        elif gridworld.is_terminal(state):
            print(f"⚠️  Estado {state} é TERMINAL - ignorado")
        else:
            valid_states.append(state)
    
    if not valid_states:
        print("❌ Nenhum estado válido selecionado!")
        return None
    
    # ========================================================================
    # ANÁLISE DOS VALORES
    # ========================================================================
    
    print("\n" + "="*80)
    print("COMPARAÇÃO DE Q-VALUES ENTRE ALGORITMOS")
    print("="*80)
    print(f"Algoritmos analisados: {list(Q_dict.keys())}")
    print(f"Número de estados: {len(valid_states)}")
    print(f"Estados: {valid_states}")
    print("="*80)
    
    # Preparar dados para DataFrame
    data_rows = []
    
    if show_actions:
        # Mostrar Q-values de todas as ações
        print("\n" + "─"*80)
        print("Q-VALUES POR ESTADO E AÇÃO")
        print("─"*80)
        
        for state in valid_states:
            state_idx = state[0] * gridworld.cols + state[1]
            
            print(f"\n{'Estado ' + str(state):^80}")
            print("─"*80)
            
            for action_idx, action in enumerate(gridworld.actions):
                row_data = {
                    'Estado': str(state),
                    'Ação': action
                }
                
                print(f"  Ação {action}:  ", end='')
                
                q_values = []
                for algo_name, Q in Q_dict.items():
                    q_val = Q[state_idx, action_idx]
                    q_values.append(q_val)
                    row_data[algo_name] = q_val
                    print(f"{algo_name}: {q_val:>8.4f}  ", end='')
                
                # Calcular diferenças se solicitado
                if show_differences and len(Q_dict) > 1:
                    max_diff = max(q_values) - min(q_values)
                    row_data['Diff_Max'] = max_diff
                    print(f"| Δmax: {max_diff:>7.4f}", end='')
                
                print()
                data_rows.append(row_data)
            
            # Mostrar melhor ação para cada algoritmo
            print(f"\n  {'Melhor ação:':<15}", end='')
            for algo_name, Q in Q_dict.items():
                best_action_idx = np.argmax(Q[state_idx])
                best_action = gridworld.actions[best_action_idx]
                print(f"{algo_name}: {best_action:<6}", end='')
            print("\n")
    
    else:
        # Mostrar apenas valores máximos V(s) = max_a Q(s,a)
        print("\n" + "─"*80)
        print("VALORES MÁXIMOS POR ESTADO: V(s) = max_a Q(s,a)")
        print("─"*80)
        print(f"\n{'Estado':<15}", end='')
        
        for algo_name in Q_dict.keys():
            print(f"{algo_name:<20}", end='')
        
        if show_differences and len(Q_dict) > 1:
            print("Δmax", end='')
        
        print("\n" + "─"*80)
        
        for state in valid_states:
            state_idx = state[0] * gridworld.cols + state[1]
            
            row_data = {'Estado': str(state)}
            
            print(f"{str(state):<15}", end='')
            
            max_values = []
            for algo_name, Q in Q_dict.items():
                max_q = np.max(Q[state_idx])
                max_values.append(max_q)
                row_data[algo_name] = max_q
                print(f"{max_q:<20.4f}", end='')
            
            if show_differences and len(Q_dict) > 1:
                max_diff = max(max_values) - min(max_values)
                row_data['Diff_Max'] = max_diff
                print(f"{max_diff:<8.4f}", end='')
            
            print()
            data_rows.append(row_data)
    
    # ========================================================================
    # ESTATÍSTICAS DE DIFERENÇAS
    # ========================================================================
    
    if show_differences and len(Q_dict) > 1:
        print("\n" + "="*80)
        print("ANÁLISE DE DIVERGÊNCIAS")
        print("="*80)
        
        differences = [row['Diff_Max'] for row in data_rows if 'Diff_Max' in row]
        
        print(f"Diferença média:   {np.mean(differences):.4f}")
        print(f"Diferença máxima:  {np.max(differences):.4f}")
        print(f"Diferença mínima:  {np.min(differences):.4f}")
        print(f"Desvio padrão:     {np.std(differences):.4f}")
        
        # Identificar estados com maiores diferenças
        if not show_actions:
            sorted_diffs = sorted(
                [(row['Estado'], row['Diff_Max']) for row in data_rows],
                key=lambda x: x[1],
                reverse=True
            )
            
            print(f"\nEstados com MAIOR divergência:")
            for i, (state, diff) in enumerate(sorted_diffs[:5], 1):
                print(f"  {i}. Estado {state}: Δ = {diff:.4f}")
            
            print(f"\nEstados com MENOR divergência:")
            for i, (state, diff) in enumerate(sorted_diffs[-5:], 1):
                print(f"  {i}. Estado {state}: Δ = {diff:.4f}")
    
    # ========================================================================
    # GRÁFICO COMPARATIVO
    # ========================================================================
    
    if plot_comparison:
        print("\n📊 Gerando gráfico comparativo...")
        
        df = pd.DataFrame(data_rows)
        
        if show_actions:
            # Gráfico com todas as ações
            fig, axes = plt.subplots(
                len(valid_states), 1, 
                figsize=(figsize[0], figsize[1] * len(valid_states) / 4),
                squeeze=False
            )
            
            for idx, state in enumerate(valid_states):
                ax = axes[idx, 0]
                
                # Filtrar dados deste estado
                state_data = df[df['Estado'] == str(state)]
                
                # Preparar dados para o gráfico
                x = np.arange(len(gridworld.actions))
                width = 0.8 / len(Q_dict)
                
                for i, algo_name in enumerate(Q_dict.keys()):
                    values = state_data[algo_name].values
                    offset = (i - len(Q_dict)/2 + 0.5) * width
                    ax.bar(x + offset, values, width, label=algo_name, alpha=0.8)
                
                ax.set_xlabel('Ação')
                ax.set_ylabel('Q-value')
                ax.set_title(f'Estado {state}', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(gridworld.actions)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        else:
            # Gráfico apenas com valores máximos
            fig, ax = plt.subplots(figsize=figsize)
            
            x = np.arange(len(valid_states))
            width = 0.8 / len(Q_dict)
            
            for i, algo_name in enumerate(Q_dict.keys()):
                values = df[algo_name].values
                offset = (i - len(Q_dict)/2 + 0.5) * width
                ax.bar(x + offset, values, width, label=algo_name, alpha=0.8)
            
            ax.set_xlabel('Estado')
            ax.set_ylabel('V(s) = max_a Q(s,a)')
            ax.set_title('Comparação de Valores Máximos por Estado', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in valid_states], rotation=45)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # RETORNAR DATAFRAME
    # ========================================================================
    
    df_result = pd.DataFrame(data_rows)
    
    print("\n" + "="*80)
    print("✅ Análise completa!")
    print("="*80)
    print(f"\nDataFrame retornado com {len(df_result)} linhas")
    print("Para acessar: df = compare_q_values_by_states(...)")
    
    return df_result


# ============================================================================
# FUNÇÕES AUXILIARES ADICIONAIS
# ============================================================================

def find_most_divergent_states(
    Q_dict: Dict[str, np.ndarray],
    gridworld: GridWorld,
    n_states: int = 10
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Identifica os estados com maior divergência entre algoritmos.
    
    Parâmetros:
    -----------
    Q_dict : Dict[str, np.ndarray]
        Dicionário de tabelas Q
    gridworld : GridWorld
        Ambiente
    n_states : int, default=10
        Número de estados a retornar
    
    Retorna:
    --------
    List[Tuple[Tuple[int, int], float]]
        Lista de (estado, divergência) ordenada por divergência
    
    Exemplo:
    --------
    >>> divergent = find_most_divergent_states(Q_dict, gw, n_states=5)
    >>> print("Estados mais divergentes:")
    >>> for state, div in divergent:
    ...     print(f"  {state}: {div:.4f}")
    """
    divergences = []
    
    for state in gridworld.states:
        if not gridworld.is_terminal(state) and state not in gridworld.walls:
            state_idx = state[0] * gridworld.cols + state[1]
            
            # Pegar valores máximos de cada algoritmo
            max_values = [np.max(Q[state_idx]) for Q in Q_dict.values()]
            
            # Calcular divergência (diferença max - min)
            divergence = max(max_values) - min(max_values)
            divergences.append((state, divergence))
    
    # Ordenar por divergência (decrescente)
    divergences.sort(key=lambda x: x[1], reverse=True)
    
    return divergences[:n_states]


def compare_convergence_by_state(
    Q_dict: Dict[str, np.ndarray],
    gridworld: GridWorld,
    state: Tuple[int, int],
    reference_algo: str = None
) -> None:
    """
    Compara como diferentes algoritmos convergiram para um estado específico.
    
    Parâmetros:
    -----------
    Q_dict : Dict[str, np.ndarray]
        Dicionário de tabelas Q
    gridworld : GridWorld
        Ambiente
    state : Tuple[int, int]
        Estado para analisar
    reference_algo : str, opcional
        Nome do algoritmo de referência para calcular diferenças.
        Se None, usa o primeiro algoritmo.
    
    Exemplo:
    --------
    >>> compare_convergence_by_state(Q_dict, gw, (1, 2), reference_algo='Q-Learning')
    """
    if state in gridworld.walls or gridworld.is_terminal(state):
        print(f"❌ Estado {state} é inválido (parede ou terminal)")
        return
    
    state_idx = state[0] * gridworld.cols + state[1]
    
    print("\n" + "="*70)
    print(f"ANÁLISE DE CONVERGÊNCIA - Estado {state}")
    print("="*70)
    
    # Escolher algoritmo de referência
    if reference_algo is None:
        reference_algo = list(Q_dict.keys())[0]
    
    if reference_algo not in Q_dict:
        print(f"❌ Algoritmo de referência '{reference_algo}' não encontrado")
        return
    
    Q_ref = Q_dict[reference_algo][state_idx]
    
    print(f"\nAlgoritmo de referência: {reference_algo}")
    print("─"*70)
    
    # Tabela comparativa
    print(f"\n{'Ação':<10}", end='')
    for algo_name in Q_dict.keys():
        print(f"{algo_name:<15}", end='')
    print(f"Δ vs {reference_algo}")
    print("─"*70)
    
    for action_idx, action in enumerate(gridworld.actions):
        print(f"{action:<10}", end='')
        
        for algo_name, Q in Q_dict.items():
            q_val = Q[state_idx, action_idx]
            print(f"{q_val:<15.4f}", end='')
        
        if reference_algo in Q_dict:
            diff = Q[state_idx, action_idx] - Q_ref[action_idx]
            print(f"{diff:>+8.4f}")
        else:
            print()
    
    print("─"*70)
    print(f"\n{'Melhor ação:':<10}", end='')
    for algo_name, Q in Q_dict.items():
        best_action = gridworld.actions[np.argmax(Q[state_idx])]
        print(f"{best_action:<15}", end='')
    print()


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("""
    EXEMPLO DE USO DA FUNÇÃO compare_q_values_by_states
    ====================================================
    
    # Supondo que você já treinou os algoritmos:
    # Q_sarsa, Q_qlearning, Q_expected = ...
    
    Q_dict = {
        'SARSA': Q_sarsa,
        'Q-Learning': Q_qlearning,
        'Expected SARSA': Q_expected
    }
    
    # 1. Comparar estados específicos com todas as ações
    estados_interesse = [(0,1), (1,1), (2,1), (3,0)]
    df = compare_q_values_by_states(
        Q_dict, gw, 
        selected_states=estados_interesse,
        show_actions=True,
        show_differences=True,
        plot_comparison=True
    )
    
    # 2. Comparar apenas valores máximos (mais simples)
    df_simple = compare_q_values_by_states(
        Q_dict, gw, 
        selected_states=estados_interesse,
        show_actions=False
    )
    
    # 3. Encontrar estados mais divergentes
    divergent_states = find_most_divergent_states(Q_dict, gw, n_states=5)
    
    # 4. Analisar convergência de um estado específico
    compare_convergence_by_state(Q_dict, gw, (1,2), reference_algo='Q-Learning')
    """)
