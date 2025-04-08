---
aliases:
  - MDP
  - Markov Decision Process
  - Markovian Decision Processes
  - Stochastic Dynamic Programming
---

# Markov Decision Processes

Markov Decision Processes (MDPs) provide a mathematical framework for modeling sequential decision-making problems under uncertainty. They are fundamental to [[Reinforcement Learning]] and serve as the theoretical underpinning for many algorithms in the field.

## Core Concepts

An MDP is formally defined as a tuple $(S, A, P, R, \gamma)$ where:

- **State Space ($S$)**: The set of all possible states of the environment
- **Action Space ($A$)**: The set of all possible actions available to the agent
- **Transition Probability ($P$)**: $P(s'|s,a)$ defines the probability of transitioning to state $s'$ when taking action $a$ in state $s$
- **Reward Function ($R$)**: $R(s,a,s')$ specifies the immediate reward received after transitioning from state $s$ to $s'$ due to action $a$
- **Discount Factor ($\gamma \in [0,1]$)**: Determines the present value of future rewards

The defining property of MDPs is the **Markov property**: the future depends only on the current state and action, not on the history of states and actions. Mathematically:

$$P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1},...,s_0, a_0) = P(s_{t+1}|s_t, a_t)$$

## Policy and Value Functions

### Policy

A **policy** $\pi$ maps states to actions, guiding the agent's behavior:

- **Deterministic Policy**: $\pi(s) = a$
- **Stochastic Policy**: $\pi(a|s)$ gives the probability of taking action $a$ in state $s$

### Value Functions

Value functions quantify the expected cumulative discounted reward from a state or state-action pair:

- **State-Value Function** ($V^\pi(s)$): Expected return when starting in state $s$ and following policy $\pi$

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s\right]$$

- **Action-Value Function** ($Q^\pi(s,a)$): Expected return when taking action $a$ in state $s$ and following policy $\pi$ thereafter

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right]$$

### Bellman Equations

The Bellman equations express value functions recursively:

- **For State-Value**:
$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

- **For Action-Value**:
$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

## Optimal Policies and Value Functions

The goal in solving an MDP is to find an optimal policy $\pi^*$ that maximizes the expected return from all states:

- **Optimal State-Value Function**:
$$V^*(s) = \max_{\pi} V^\pi(s) = \max_{a} \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$$

- **Optimal Action-Value Function**:
$$Q^*(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

- **Optimal Policy**:
$$\pi^*(s) = \arg\max_{a} Q^*(s,a)$$

## Solving MDPs

Several methods exist for finding optimal policies in MDPs:

### Dynamic Programming Methods

- **[[Value Iteration]]**: Iteratively computes the optimal value function by applying Bellman optimality equations:
  ```python
  def value_iteration(mdp, theta=0.001, gamma=0.9):
      V = {s: 0 for s in mdp.states}
      while True:
          delta = 0
          for s in mdp.states:
              v = V[s]
              V[s] = max([sum([p * (mdp.R(s, a, s_prime) + gamma * V[s_prime]) 
                      for s_prime, p in mdp.transitions(s, a).items()])
                      for a in mdp.actions(s)])
              delta = max(delta, abs(v - V[s]))
          if delta < theta:
              break
      
      # Extract policy
      policy = {s: argmax([sum([p * (mdp.R(s, a, s_prime) + gamma * V[s_prime]) 
                      for s_prime, p in mdp.transitions(s, a).items()])
                      for a in mdp.actions(s)])
               for s in mdp.states}
      return policy, V
  ```

- **[[Policy Iteration]]**: Alternates between policy evaluation and improvement:
  ```python
  def policy_iteration(mdp, gamma=0.9):
      # Initialize policy randomly
      policy = {s: random.choice(mdp.actions(s)) for s in mdp.states}
      
      while True:
          # Policy evaluation
          V = policy_evaluation(policy, mdp, gamma)
          
          # Policy improvement
          stable = True
          for s in mdp.states:
              old_action = policy[s]
              policy[s] = argmax([sum([p * (mdp.R(s, a, s_prime) + gamma * V[s_prime]) 
                         for s_prime, p in mdp.transitions(s, a).items()])
                         for a in mdp.actions(s)])
              if old_action != policy[s]:
                  stable = False
          
          if stable:
              return policy, V
  ```

- **Modified Policy Iteration**: Combines elements of both value and policy iteration for improved efficiency

### Model-Free Methods

When the transition probabilities and rewards are unknown, model-free methods can be used:

- **[[Monte Carlo Methods]]**: Estimate value functions from experience
- **[[Temporal Difference Learning]]**: Update value estimates based on other value estimates
  - **Q-Learning**: Off-policy TD method that directly learns the optimal action-value function
  - **SARSA**: On-policy TD method that learns the action-value function for the current policy

## Extensions and Variants

### Partially Observable MDPs (POMDPs)

When states cannot be directly observed, the problem becomes a [[Partially Observable MDP]] (POMDP). In POMDPs:
- The agent receives observations that provide incomplete information about the state
- The agent must maintain a belief state (probability distribution over possible states)
- Solving POMDPs is significantly more complex than solving MDPs

### Constrained MDPs (CMDPs)

[[Constrained Markov Decision Processes]] extend MDPs by adding constraints on certain behaviors or outcomes:
- Multiple cost functions in addition to the reward function
- Constraints on the expected cumulative cost
- Solved using linear programming or Lagrangian methods
- Applications include safety-critical systems, resource-constrained environments, and motion planning in robotics

### Continuous-Time MDPs

[[Continuous-Time Markov Decision Processes]] model decision problems in continuous time:
- States evolve continuously over time according to a controlled continuous-time Markov process
- Actions can be taken at any point in time
- Solved using methods based on Hamilton-Jacobi-Bellman equations or linear programming

## Applications

MDPs have been applied to a wide range of domains:

- **Robotics**: Path planning, control, and navigation
- **Finance**: Portfolio management, option pricing, and risk management
- **Healthcare**: Treatment planning and resource allocation
- **Telecommunications**: Network routing and resource allocation
- **Energy**: Smart grid management and renewable energy integration
- **Recommendation Systems**: Sequential recommendation problems
- **Game AI**: Strategy development in games with uncertainty

## Connection to Reinforcement Learning

MDPs form the theoretical foundation of [[Reinforcement Learning]]:

- Model-based RL methods explicitly learn the MDP parameters (transition probabilities and rewards)
- Model-free RL methods learn value functions or policies directly without explicitly modeling the environment
- Deep RL methods use neural networks to represent value functions or policies for high-dimensional MDPs

## Tools and Libraries

Several libraries exist for working with MDPs:

- **PyMDPToolbox**: Python library for solving MDPs
- **MDP Toolbox**: MATLAB/Octave library for MDP algorithms
- **OpenAI Gym**: Provides environments that follow the MDP framework
- **RLlib**: Scalable library for reinforcement learning with support for various MDP solvers

## Conclusion

Markov Decision Processes provide a powerful framework for sequential decision-making under uncertainty. Their mathematical foundations underpin many algorithms in reinforcement learning and stochastic control. Understanding MDPs is essential for developing effective algorithms for complex decision problems across various domains.

---

**References**:
1. Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming. Wiley.
2. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). The MIT Press.
3. Bellman, R. (1957). Dynamic Programming. Princeton University Press. 