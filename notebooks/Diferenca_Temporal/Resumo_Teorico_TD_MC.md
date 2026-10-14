# Algoritmos de Monte Carlo e Diferença Temporal em Aprendizado por Reforço

## 1. Introdução ao Aprendizado por Reforço

### Contexto e Definição

O **Aprendizado por Reforço (RL)** é um paradigma de aprendizado de máquina onde um agente aprende a tomar decisões através da interação com um ambiente, recebendo recompensas ou penalidades baseadas em suas ações. O objetivo fundamental é aprender uma política ótima que maximize a recompensa cumulativa esperada ao longo do tempo.

### Processos de Decisão de Markov (MDPs)

Um **MDP** é uma formalização matemática para tomada de decisão sequencial, definido pela tupla $(S, A, P, R, \gamma)$:

- **$S$**: Conjunto finito de estados
- **$A$**: Conjunto finito de ações
- **$P$**: Função de transição $P(s'|s,a)$ - probabilidade de transição para estado $s'$ dado estado $s$ e ação $a$
- **$R$**: Função de recompensa $R(s,a,s')$ - recompensa imediata
- **$\gamma \in [0,1]$**: Fator de desconto para recompensas futuras

### Função de Valor e Política

- **Política** $\pi(a|s)$: Distribuição de probabilidade sobre ações dado um estado
- **Função Valor de Estado** $V^\pi(s)$: Valor esperado começando em $s$ e seguindo política $\pi$
- **Função Valor de Ação** $Q^\pi(s,a)$: Valor esperado tomando ação $a$ em $s$ e depois seguindo $\pi$

---

## 2. Conceitos Fundamentais

### Estados e Ações

- **Estado ($s \in S$)**: Representação completa da situação atual do agente no ambiente
- **Ação ($a \in A$)**: Decisão que o agente pode tomar em cada estado

### Recompensas e Retornos

- **Recompensa Imediata** $R_{t+1}$: Feedback escalar recebido após executar ação $A_t$ no estado $S_t$
- **Retorno** $G_t$: Soma descontada de recompensas futuras

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### Função Valor de Estado V(s)

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s\right]$$

**Interpretação**: Retorno esperado começando no estado $s$ e seguindo a política $\pi$.

### Função Valor de Ação Q(s,a)

$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s, A_t = a\right]$$

**Interpretação**: Retorno esperado tomando ação $a$ no estado $s$ e depois seguindo política $\pi$.

### Política π(a|s)

$$\pi(a|s) = P(A_t = a | S_t = s)$$

Uma política **determinística** pode ser escrita como $\pi(s) = a$, enquanto uma política **estocástica** define uma distribuição de probabilidade sobre as ações.

---

## 3. Parâmetros Principais

### γ (Gamma) - Fator de Desconto

- **Propósito**: Pondera a importância de recompensas futuras versus imediatas
- **Intervalo**: $\gamma \in [0, 1]$
- **Efeitos**:
  - $\gamma = 0$: Considera apenas recompensa imediata (míope)
  - $\gamma \rightarrow 1$: Considera recompensas futuras quase igualmente importantes
  - $\gamma = 1$: Sem desconto (apenas para tarefas episódicas finitas)
- **Valores típicos**: 0.9 - 0.99

### α (Alpha) - Taxa de Aprendizado

- **Propósito**: Controla quanto das novas informações é incorporado
- **Intervalo**: $\alpha \in (0, 1]$
- **Fórmula de atualização genérica**:

$$\text{NovoValor} \leftarrow (1-\alpha) \cdot \text{ValorAntigo} + \alpha \cdot \text{NovaEstimativa}$$

- **Efeitos**:
  - $\alpha$ pequeno (0.01-0.1): Aprendizado lento mas estável
  - $\alpha$ grande (0.5-1.0): Aprendizado rápido mas potencialmente instável
- **Valores típicos**: 0.1 - 0.5

### ε (Epsilon) - Exploração vs Explotação

- **Propósito**: Balancea exploração de novas ações versus uso do conhecimento atual
- **Intervalo**: $\varepsilon \in [0, 1]$
- **Política $\varepsilon$-greedy**:

$$\pi(a|s) = \begin{cases}
1 - \varepsilon + \frac{\varepsilon}{|A|} & \text{se } a = \arg\max_{a'} Q(s,a') \\
\frac{\varepsilon}{|A|} & \text{caso contrário}
\end{cases}$$

- **Efeitos**:
  - $\varepsilon = 0$: Puramente guloso (sem exploração)
  - $\varepsilon = 1$: Puramente aleatório
- **Valores típicos**: 0.1 - 0.3 (com possível decaimento ao longo do tempo)

---

## 4. Métodos de Monte Carlo

### 4.1 Princípios Fundamentais

Os métodos de Monte Carlo (MC) aprendem diretamente da experiência **completa** em episódios. Eles:
- **Não requerem modelo** do ambiente (model-free)
- Usam **amostragem** para estimar valores esperados
- Atualizam valores apenas ao **final de episódios**
- Baseiam-se na **Lei dos Grandes Números**: média de amostras converge para valor esperado

**Ideia Central**: 
$$V(s) \approx \frac{1}{N} \sum_{i=1}^{N} G_i(s)$$

onde $G_i(s)$ é o retorno observado na $i$-ésima visita ao estado $s$.

### 4.2 First-Visit vs Every-Visit

#### First-Visit Monte Carlo
- Considera apenas a **primeira visita** a cada estado em um episódio
- Estimativas são médias de retornos de primeiras visitas
- Propriedade: estimativas são não-viesadas

#### Every-Visit Monte Carlo
- Considera **todas as visitas** a cada estado em um episódio
- Estimativas são médias de todos os retornos observados
- Converge assintoticamente mas pode ter viés inicial

### 4.3 Predição de Monte Carlo

**Objetivo**: Estimar $V^\pi$ para uma política fixa $\pi$.

**Algoritmo (First-Visit MC Prediction)**:

```
Inicializar:
  V(s) ← arbitrário, ∀s ∈ S
  Returns(s) ← lista vazia, ∀s ∈ S

Para cada episódio:
  Gerar episódio seguindo π: S₀, A₀, R₁, S₁, A₁, R₂, ..., S_T
  G ← 0
  Para t = T-1, T-2, ..., 0:
    G ← γG + R_{t+1}
    Se S_t não aparece em S₀, S₁, ..., S_{t-1}:
      Adicionar G a Returns(S_t)
      V(S_t) ← média(Returns(S_t))
```

### 4.4 Controle de Monte Carlo

**Objetivo**: Encontrar política ótima $\pi^*$.

**Generalização da Iteração de Política (GPI)**:
1. **Avaliação**: Estimar $Q^\pi$ para política atual
2. **Melhoria**: Tornar política gulosa em relação a $Q$

### 4.5 Exploring Starts

Para garantir exploração adequada, Monte Carlo com **Exploring Starts** assume que cada episódio começa com um par estado-ação $(s,a)$ escolhido aleatoriamente.

**Algoritmo (Monte Carlo ES)**:

```
Inicializar:
  π(s) ∈ A(s) arbitrário, ∀s ∈ S
  Q(s,a) ∈ ℝ arbitrário, ∀s ∈ S, a ∈ A(s)
  Returns(s,a) ← lista vazia, ∀s ∈ S, a ∈ A(s)

Repetir (para cada episódio):
  Escolher S₀ ∈ S, A₀ ∈ A(S₀) aleatoriamente (exploring starts)
  Gerar episódio a partir de S₀, A₀ seguindo π
  G ← 0
  Para cada passo t = T-1, T-2, ..., 0:
    G ← γG + R_{t+1}
    Se par (S_t, A_t) não aparece antes no episódio:
      Adicionar G a Returns(S_t, A_t)
      Q(S_t, A_t) ← média(Returns(S_t, A_t))
      π(S_t) ← argmax_a Q(S_t, a)
```

### 4.6 Fórmulas Matemáticas

**Atualização Incremental de Monte Carlo**:

$$V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)]$$

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[G_t - Q(S_t, A_t)]$$

onde $G_t$ é o retorno observado após visitar $(S_t, A_t)$.

### 4.7 Pseudocódigo

**Monte Carlo com política $\varepsilon$-greedy**:

```
Algoritmo: Monte Carlo Control (ε-greedy)
===========================================
Inicializar:
  Q(s,a) ← arbitrário, ∀s,a
  π ← política ε-greedy baseada em Q
  
Para cada episódio:
  Gerar episódio usando π: S₀,A₀,R₁,...,S_T
  G ← 0
  Para t = T-1, T-2, ..., 0:
    G ← γG + R_{t+1}
    Q(S_t,A_t) ← Q(S_t,A_t) + α[G - Q(S_t,A_t)]
    π ← ε-greedy(Q)
```

### 4.8 Vantagens e Limitações

**Vantagens**:
- ✓ Não precisa de modelo do ambiente
- ✓ Simples e intuitivo
- ✓ Converge para valores verdadeiros
- ✓ Funciona bem com recompensas esparsas
- ✓ Pode focar em estados de interesse

**Limitações**:
- ✗ Requer episódios completos
- ✗ Alta variância nas estimativas
- ✗ Convergência lenta
- ✗ Não funciona para tarefas contínuas
- ✗ Ineficiente em episódios longos

---

## 5. Métodos de Diferença Temporal (TD)

### 5.1 Princípios Fundamentais

Métodos TD combinam ideias de **Monte Carlo** e **Programação Dinâmica**:
- Como MC: aprendem **diretamente da experiência** (model-free)
- Como DP: fazem **bootstrapping** - atualizam estimativas baseadas em outras estimativas
- Atualizam valores **a cada passo**, não apenas ao final do episódio

**Ideia Central - TD(0)**:

$$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

O termo $[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$ é chamado **erro TD** ou **TD error**.

#### On-Policy vs Off-Policy

**On-Policy**:
- Aprende sobre a política que está **seguindo**
- Avalia e melhora a mesma política usada para gerar dados
- Exemplo: SARSA

**Off-Policy**:
- Aprende sobre uma política **diferente** da que está seguindo
- Política alvo (target) ≠ Política comportamental (behavior)
- Exemplo: Q-Learning

### 5.2 TD(0) - Predição

**Objetivo**: Estimar $V^\pi$ para política fixa $\pi$.

**Algoritmo TD(0)**:

```
Inicializar:
  V(s) ← arbitrário, ∀s ∈ S
  V(terminal) ← 0

Para cada episódio:
  Inicializar S
  Para cada passo do episódio:
    A ← ação dada por π para S
    Executar A, observar R, S'
    V(S) ← V(S) + α[R + γV(S') - V(S)]
    S ← S'
  Até S ser terminal
```

**Propriedades**:
- Atualização online (a cada passo)
- Bootstrapping: usa $V(S_{t+1})$ para estimar $V(S_t)$
- Converge para $V^\pi$ sob certas condições

### 5.3 SARSA (On-Policy)

**State-Action-Reward-State-Action**

**Equação de Atualização**:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

**Características**:
- Usa a ação $A_{t+1}$ que **será realmente executada**
- Aprende valores considerando a exploração
- Comportamento mais **cauteloso** e seguro

**Pseudocódigo SARSA**:

```
Algoritmo: SARSA
=================
Inicializar Q(s,a) arbitrariamente
Parâmetros: α, γ, ε

Para cada episódio:
  Inicializar S
  Escolher A de S usando política ε-greedy(Q)
  
  Para cada passo do episódio:
    Executar ação A, observar R, S'
    Escolher A' de S' usando política ε-greedy(Q)
    Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
    S ← S'; A ← A'
  Até S ser terminal
```

### 5.4 Q-Learning (Off-Policy)

**Equação de Atualização**:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

**Características**:
- Usa o **máximo** valor Q no próximo estado
- Aprende política **ótima** independentemente da política de exploração
- Mais **agressivo** na busca por otimalidade

**Pseudocódigo Q-Learning**:

```
Algoritmo: Q-Learning
======================
Inicializar Q(s,a) arbitrariamente
Parâmetros: α, γ, ε

Para cada episódio:
  Inicializar S
  
  Para cada passo do episódio:
    Escolher A de S usando política ε-greedy(Q)
    Executar ação A, observar R, S'
    Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
    S ← S'
  Até S ser terminal
```

### 5.5 Expected SARSA

**Equação de Atualização**:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \mathbb{E}_\pi[Q(S_{t+1}, A_{t+1})] - Q(S_t, A_t)]$$

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \sum_a \pi(a|S_{t+1})Q(S_{t+1}, a) - Q(S_t, A_t)]$$

**Características**:
- Usa **expectativa** sobre as ações possíveis
- **Elimina variância** da seleção de ação
- Performance geralmente **superior** a SARSA e Q-Learning

**Pseudocódigo Expected SARSA**:

```
Algoritmo: Expected SARSA
==========================
Inicializar Q(s,a) arbitrariamente
Parâmetros: α, γ, ε

Para cada episódio:
  Inicializar S
  
  Para cada passo do episódio:
    Escolher A de S usando política ε-greedy(Q)
    Executar ação A, observar R, S'
    
    // Calcular expectativa
    V_expected ← 0
    Para cada ação a em S':
      π(a|S') ← probabilidade de escolher a (ε-greedy)
      V_expected ← V_expected + π(a|S') * Q(S',a)
    
    Q(S,A) ← Q(S,A) + α[R + γ * V_expected - Q(S,A)]
    S ← S'
  Até S ser terminal
```

### 5.6 Fórmulas Matemáticas

**Erro TD (Temporal Difference Error)**:

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

**Relação com Monte Carlo**:

$$G_t - V(S_t) = \sum_{k=0}^{T-t-1} \gamma^k \delta_{t+k}$$

Ou seja, o erro de Monte Carlo é a soma dos erros TD!

### 5.7 Pseudocódigos Comparativos

```
TD(0) vs Monte Carlo
====================

Monte Carlo:                     TD(0):
-----------                       -----
Espera fim do episódio           Atualiza a cada passo
V(S_t) ← V(S_t) + α[G_t - V(S_t)] V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
Usa retorno real G_t             Usa estimativa V(S_{t+1})
```

### 5.8 Vantagens e Limitações

**Vantagens TD**:
- ✓ Aprendizado online (não espera fim do episódio)
- ✓ Funciona em tarefas contínuas
- ✓ Menor variância que Monte Carlo
- ✓ Mais eficiente em episódios longos
- ✓ Bootstrapping acelera aprendizado

**Limitações TD**:
- ✗ Bootstrapping introduz viés
- ✗ Convergência mais sensível a parâmetros
- ✗ Pode propagar erros de estimação
- ✗ Mais complexo de analisar teoricamente

---

## 6. Análise Comparativa

### 6.1 Diferenças Conceituais

| Aspecto | Monte Carlo | TD (SARSA/Q-Learning) |
|---------|-------------|----------------------|
| **Atualização** | Final do episódio | A cada passo |
| **Alvo** | Retorno real $G_t$ | Estimativa bootstrapped |
| **Viés** | Não-viesado | Viesado (bootstrapping) |
| **Variância** | Alta | Baixa |
| **Episódios** | Requer término | Funciona em contínuo |
| **Modelo** | Model-free | Model-free |

### 6.2 Diferenças Algorítmicas

| Característica | SARSA | Q-Learning | Expected SARSA |
|----------------|-------|------------|----------------|
| **Tipo** | On-policy | Off-policy | Ambos |
| **Atualização** | $Q(S',A')$ | $\max_a Q(S',a)$ | $\mathbb{E}[Q(S',a)]$ |
| **Comportamento** | Cauteloso | Otimista | Balanceado |
| **Exploração afeta valores** | Sim | Não | Ponderado |
| **Complexidade** | $O(1)$ | $O(|A|)$ | $O(|A|)$ |

### 6.3 Convergência

**Monte Carlo**:
- Converge para $V^\pi$ ou $Q^\pi$ com visitas infinitas
- Taxa: $O(1/\sqrt{n})$ onde $n$ é número de visitas

**TD (SARSA)**:
- Converge para $Q^\pi$ (política sendo seguida)
- Requer condições de Robbins-Monro para $\alpha$

**Q-Learning**:
- Converge para $Q^*$ (política ótima)
- Independente da política de exploração

**Expected SARSA**:
- Converge mais rapidamente que SARSA
- Pode convergir para $Q^*$ quando $\varepsilon \rightarrow 0$

### 6.4 Viés vs Variância

```
Trade-off Fundamental:
=====================
         Viés ↓        Variância ↑
MC: ----------------→ Retorno Real
         Alta Variância, Sem Viés

         Viés ↑        Variância ↓  
TD: ----------------→ Bootstrap
         Baixa Variância, Com Viés
```

### 6.5 Eficiência Computacional

| Método | Memória | Tempo/Atualização | Tempo/Episódio |
|--------|---------|-------------------|----------------|
| Monte Carlo | $O(T)$ buffer | $O(1)$ | $O(T)$ |
| TD(0) | $O(1)$ | $O(1)$ | $O(T)$ |
| SARSA | $O(|S| \times |A|)$ | $O(1)$ | $O(T)$ |
| Q-Learning | $O(|S| \times |A|)$ | $O(|A|)$ | $O(T \times |A|)$ |
| Expected SARSA | $O(|S| \times |A|)$ | $O(|A|)$ | $O(T \times |A|)$ |

### 6.6 Casos de Uso

**Use Monte Carlo quando**:
- Episódios são curtos
- Precisão é mais importante que velocidade
- Recompensas muito esparsas
- Quer estimativas não-viesadas

**Use TD/SARSA quando**:
- Segurança durante treinamento é crítica
- Ambiente tem riscos/penalidades
- Episódios são longos ou contínuos

**Use Q-Learning quando**:
- Objetivo é política ótima
- Pode tolerar exploração arriscada
- Quer garantias de otimalidade

**Use Expected SARSA quando**:
- Quer melhor performance geral
- Estabilidade é importante
- Pode arcar com custo computacional extra

---

## 7. Exemplos Práticos

### GridWorld 4x3

Grid simples com estados terminais e recompensas:

```
┌───┬───┬───┬───┐
│ S │   │   │ +1│  (+1: objetivo)
├───┼───┼───┼───┤
│   │ ■ │   │ -1│  (-1: penalidade, ■: obstáculo)
├───┼───┼───┼───┤
│   │   │   │   │  (S: início)
└───┴───┴───┴───┘
```

**Comportamento dos Algoritmos**:
- **Monte Carlo**: Aprende após completar episódios inteiros
- **SARSA**: Aprende caminho seguro, evitando -1
- **Q-Learning**: Aprende caminho ótimo mais direto

### Cliff Walking

Ambiente clássico para comparar algoritmos:

```
┌─────────────────────────────────┐
│ S □ □ □ □ □ □ □ □ □ □ □ □ □ □ G │  (Caminho seguro)
├─────────────────────────────────┤
│ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ │  (Precipício: -100)
└─────────────────────────────────┘
S: início, G: objetivo, █: cliff
```

**Resultados Típicos**:
- **SARSA**: Aprende caminho superior (seguro)
  - Recompensa média: ~-20 por episódio
  - Evita o precipício considerando exploração

- **Q-Learning**: Aprende caminho beirando precipício (ótimo)
  - Recompensa ótima: ~-13 por episódio
  - Mas cai frequentemente durante treinamento

- **Expected SARSA**: Melhor dos dois mundos
  - Performance online superior
  - Converge para política ótima

**Exemplo de Trajetórias Aprendidas**:

```
SARSA (Seguro):
S → → → → → → → → → → → → → → ↓
                              G

Q-Learning (Ótimo mas arriscado):
S ↓
  → → → → → → → → → → → → → → G
```

---

## 8. Considerações Finais

### Quando Usar Cada Método

**Escolha Monte Carlo se**:
- Você tem episódios bem definidos e relativamente curtos
- Precisa de estimativas não-viesadas
- Recompensas são muito esparsas (apenas no final)
- Pode esperar o fim do episódio para aprender

**Escolha TD (SARSA/Q-Learning) se**:
- Precisa de aprendizado online
- Episódios são longos ou infinitos
- Quer convergência mais rápida
- Bootstrapping não é problema

**Escolha específica entre TD**:
- **SARSA**: Quando segurança importa
- **Q-Learning**: Quando quer garantia de otimalidade
- **Expected SARSA**: Para melhor performance geral

### Combinações Possíveis

1. **TD(λ)**: Combina TD e Monte Carlo com eligibility traces
2. **n-step TD**: Meio termo entre TD(0) e Monte Carlo
3. **Double Q-Learning**: Reduz maximization bias
4. **Monte Carlo Tree Search**: Combina planning e sampling

### Desenvolvimentos Modernos

1. **Deep Reinforcement Learning**:
   - DQN usa Q-Learning com redes neurais
   - A3C combina TD com policy gradients
   - PPO usa vantagens TD para policy optimization

2. **Aproximação de Função**:
   - Substitui tabelas por aproximadores (redes neurais)
   - Permite lidar com espaços contínuos/grandes

3. **Métodos Actor-Critic**:
   - Actor: política (escolhe ações)
   - Critic: função valor (avalia ações) - usa TD!

4. **Rainbow DQN**:
   - Combina várias melhorias do Q-Learning
   - Double DQN, Prioritized Replay, Dueling Networks, etc.

---

## Referências Bibliográficas

1. **Sutton, R. S., & Barto, A. G. (2020).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
   - Capítulo 5: Monte Carlo Methods
   - Capítulo 6: Temporal-Difference Learning

2. **Russell, S., & Norvig, P. (2021).** *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
   - Capítulo 22: Reinforcement Learning

3. **Watkins, C. J. C. H. (1989).** *Learning from Delayed Rewards*. PhD Thesis, Cambridge University.
   - Introdução do Q-Learning

4. **Van Seijen, H., et al. (2009).** *A theoretical and empirical analysis of Expected Sarsa*. IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning.

5. **Mnih, V., et al. (2015).** *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529-533.
   - Deep Q-Networks (DQN)

6. **Silver, D., et al. (2016).** *Mastering the game of Go with deep neural networks and tree search*. Nature, 529(7587), 484-489.
   - AlphaGo e aplicações modernas

---

## Apêndice: Notação Matemática

| Símbolo | Significado |
|---------|-------------|
| $S$ | Conjunto de estados |
| $A$ | Conjunto de ações |
| $s, s'$ | Estado atual e próximo estado |
| $a$ | Ação |
| $\pi$ | Política |
| $\pi^*$ | Política ótima |
| $R_t$ | Recompensa no tempo $t$ |
| $G_t$ | Retorno (soma de recompensas) desde $t$ |
| $V(s)$ | Função valor de estado |
| $Q(s,a)$ | Função valor de ação |
| $\gamma$ | Fator de desconto |
| $\alpha$ | Taxa de aprendizado |
| $\varepsilon$ | Taxa de exploração |
| $\mathbb{E}$ | Valor esperado |
| $P(s'|s,a)$ | Probabilidade de transição |

---

**Fim do Resumo**

*Este material fornece uma visão abrangente dos algoritmos fundamentais de Aprendizado por Reforço, servindo como guia de estudo e referência para implementação prática.*
