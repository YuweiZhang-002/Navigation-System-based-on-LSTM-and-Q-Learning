"""
rl_agent.py
-----------
Q-learning training loop and path-visit counter utility.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------

def q_learning_train(env,
                     queue_maps,
                     mean_maps,
                     var_maps,
                     lstm_maps,
                     alpha=0.1,
                     beta=0.9,
                     gamma=0.2,
                     lamda=0.1,
                     start_loc=0,
                     end_loc=11,
                     epochs=100,
                     max_steps_per_episode=100,
                     epsilon_start=1.0,
                     epsilon_min=0.05,
                     epsilon_decay=0.995):
    """Train a Q-learning agent on the given Table environment.

    Parameters
    ----------
    env                   : Table  road-network environment instance
    queue_maps            : np.ndarray (N, 4)  current queue length per
                            (intersection, direction)
    mean_maps             : np.ndarray (N, 4)  historical mean throughput
    var_maps              : np.ndarray (N, 4)  historical throughput variance
    lstm_maps             : np.ndarray (N, 4, T)  LSTM future-flow predictions;
                            lstm_maps[state, action] returns a 1-D array of
                            length T (look-back steps)
    alpha                 : float  distance-penalty weight
    beta                  : float  congestion discount factor
    gamma                 : float  Q-learning discount factor
    lamda                 : float  learning rate (Bellman update step size)
    start_loc             : int    flat node index for episode start
    end_loc               : int    flat node index for episode goal
    epochs                : int    number of training episodes
    max_steps_per_episode : int    step limit per episode
    epsilon_start         : float  initial ε-greedy exploration rate
    epsilon_min           : float  minimum exploration rate
    epsilon_decay         : float  per-episode multiplicative decay of ε

    Returns
    -------
    q_table   : np.ndarray (block_len, act_len)  learned Q-table
    rewards   : np.ndarray (epochs,)             total reward per episode
    path_tot  : np.ndarray  node-index sequence from the final episode
    path_dir  : np.ndarray  action sequence from the final episode
    """
    q_table  = np.zeros((env.block_len(), env.act_len))
    epsilon  = epsilon_start
    rewards_tot = []
    path_tot    = [start_loc]
    path_dir    = []

    for i_episode in range(epochs):
        state         = env.reset(start_loc, end_loc)
        episode_reward = 0.0

        for _ in range(max_steps_per_episode):
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(len(env.actions))
            else:
                action = np.argmax(q_table[state])

            # Traffic inputs for current (state, action) pair
            queue    = queue_maps[state, action]
            overflow = mean_maps[state, action]
            variance = var_maps[state, action]

            # Step 1: congestion penalty (uses LSTM multi-step prediction)
            penalty_value = env.penalty_cong(
                beta,
                [overflow, variance],
                queue,
                lstm_maps[state, action],
            )

            # Step 2: environment transition
            next_state, reward, done = env.step(alpha, action, penalty_value)

            # Bellman update
            best_future_q = np.max(q_table[next_state])
            target        = reward + (gamma * best_future_q if not done else 0.0)
            q_table[state, action] += lamda * (target - q_table[state, action])

            state          = next_state
            episode_reward += reward

            # Record trajectory only in the final episode
            if i_episode == epochs - 1:
                path_tot.append(next_state)
                path_dir.append(action)

            if done:
                break

        rewards_tot.append(episode_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    path_dir.append(None)  # pad: no action after reaching goal
    return q_table, np.array(rewards_tot), np.array(path_tot), np.array(path_dir)


# ---------------------------------------------------------------------------
# Path-visit counter (diagnostic utility)
# ---------------------------------------------------------------------------

def plot_paths(visit_map, agent_locs):
    """Increment a visit-count map for each node visited by the agent.

    This is a diagnostic tool to check whether the agent backtracks
    (nodes visited more than once indicate oscillation).

    Parameters
    ----------
    visit_map  : np.ndarray  2-D array of visit counts (modified in place)
    agent_locs : iterable    sequence of flat node indices
    """
    for node in agent_locs:
        visit_map[node] += 1
