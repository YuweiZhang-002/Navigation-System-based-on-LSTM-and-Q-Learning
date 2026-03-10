"""
visualization.py
----------------
Reward-curve plotting, greedy-trajectory inference, and animated policy
visualisation for the Table road-network environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from IPython.display import HTML, display


# ---------------------------------------------------------------------------
# Reward curve
# ---------------------------------------------------------------------------

def plot_rewards(total_rewards, window=20):
    """Plot per-episode rewards and an optional moving average.

    The Y axis is fixed to [-200, 200] so that curves from different
    runs / learning rates can be compared on the same scale.

    Parameters
    ----------
    total_rewards : array-like  scalar reward for each training episode
    window        : int         moving-average window size (episodes)
    """
    plt.figure(figsize=(8, 4))
    plt.plot(total_rewards, alpha=0.7, label="Episode reward")

    if len(total_rewards) >= window:
        moving_avg = np.convolve(
            total_rewards, np.ones(window) / window, mode="valid"
        )
        plt.plot(
            range(window - 1, len(total_rewards)),
            moving_avg,
            linewidth=2,
            label=f"{window}-episode moving avg",
        )

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.ylim(-200, 200)
    plt.title("Q-learning: Episode Rewards")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Greedy trajectory
# ---------------------------------------------------------------------------

def compute_greedy_trajectory(env, start_pos, q_table,
                               end_pos=None, max_steps=100):
    """Run one greedy episode and record the agent's (row, col) positions.

    No congestion penalty is applied (penalty=0) so the trajectory reflects
    the learned spatial policy rather than stochastic traffic conditions.

    Parameters
    ----------
    env       : Table   road-network environment
    start_pos : int     flat node index for the start
    q_table   : np.ndarray (N, 4)  learned Q-table
    end_pos   : int or None  flat node index for the goal
    max_steps : int     step limit to prevent infinite loops

    Returns
    -------
    positions    : list of (row, col) tuples  agent path
    total_reward : float  accumulated reward along the greedy path
    done         : bool   True if the goal was reached
    """
    state      = env.reset(start_pos, end_pos)
    positions  = [env.get_coord(state)]
    total_reward = 0.0
    done = False

    for _ in range(max_steps):
        action = np.argmax(q_table[state])
        next_state, reward, done = env.step(alpha=0, action=action, penalty=0)

        total_reward += reward
        positions.append(env.agent_pos)
        state = next_state

        if done:
            break

    return positions, total_reward, done


# ---------------------------------------------------------------------------
# Animated policy visualisation
# ---------------------------------------------------------------------------

def animate_learned_policy_pretty(env, start_pos, q_table,
                                   end_pos=None, max_steps=100, interval=400):
    """Animate the greedy policy as a moving agent on the road-network grid.

    Renders inline in a Jupyter / Colab notebook via ``IPython.display``.

    Colour scheme
    -------------
    Light grey  (#eeeeee) : open road node
    Dark grey   (#333333) : blocked node
    Blue        (#2196f3) : start node
    Green       (#4caf50) : goal node
    Orange circle         : agent

    Parameters
    ----------
    env       : Table   road-network environment
    start_pos : int     flat node index for the start
    q_table   : np.ndarray (N, 4)  learned Q-table
    end_pos   : int or None  flat node index for the goal
    max_steps : int     step limit
    interval  : int     milliseconds between animation frames

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    positions, total_reward, done = compute_greedy_trajectory(
        env, start_pos, q_table, end_pos=end_pos, max_steps=max_steps
    )

    # Build visual grid -------------------------------------------------------
    vis_grid = env.streets.copy().astype(float)

    start_r, start_c = env.get_coord(start_pos)
    vis_grid[start_r, start_c] = 2                       # start → blue

    if env.goal_pos is not None:
        gr, gc = env.goal_pos
        vis_grid[gr, gc] = 3                             # goal  → green

    cmap = ListedColormap([
        "#eeeeee",   # 0  open road
        "#333333",   # 1  blocked
        "#2196f3",   # 2  start  (blue)
        "#4caf50",   # 3  goal   (green)
    ])

    # Create figure -----------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(max(4, env.x_range), max(3, env.y_range))
    )
    ax.set_title(
        f"Greedy trajectory  —  total reward = {total_reward:.2f},  "
        f"reached goal = {done}"
    )
    ax.imshow(vis_grid, cmap=cmap, origin="upper", vmin=0, vmax=3)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, env.x_range, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.y_range, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.8)

    # Axis labels (column / row indices)
    ax.set_xticks(np.arange(env.x_range))
    ax.set_yticks(np.arange(env.y_range))
    ax.set_xticklabels([str(c) for c in range(env.x_range)])
    ax.set_yticklabels([str(r) for r in range(env.y_range)])
    ax.invert_yaxis()

    # Agent marker
    r0, c0 = positions[0]
    robot, = ax.plot(c0, r0, "o",
                     markersize=18,
                     markeredgecolor="black",
                     markerfacecolor="orange")

    def _init():
        robot.set_data([c0], [r0])
        return (robot,)

    def _update(frame):
        r, c = positions[frame]
        robot.set_data([c], [r])
        return (robot,)

    anim = animation.FuncAnimation(
        fig, _update,
        frames=len(positions),
        init_func=_init,
        blit=True,
        interval=interval,
        repeat=False,
    )

    display(HTML(anim.to_jshtml()))
    plt.close(fig)
    return anim
