"""
environment.py
--------------
Defines the grid road-network environment (Table) and the congestion-level
mapping used by the Q-learning reward function.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Congestion-level mapping
# ---------------------------------------------------------------------------

def get_state_index(vehicle_count):
    """Map a raw vehicle count to a discrete congestion level (0–20).

    Level 0  : free flow (0 vehicles)
    Level 1–19: one level per 5 vehicles (1–95 vehicles)
    Level 20 : extremely congested (>100 vehicles)
    """
    if vehicle_count <= 0:
        level = 0
    elif vehicle_count <= 100:
        level = int(vehicle_count) // 5
    else:
        level = 20
    return level


# ---------------------------------------------------------------------------
# Road-network environment
# ---------------------------------------------------------------------------

class Table:
    """Q-table environment modelling a rectangular grid road network.

    The grid is laid out as (y_range × x_range) nodes.  Node indices are
    row-major: node k occupies row (k // x_range), column (k % x_range).

    Actions
    -------
    0 : move North  (-1,  0)
    1 : move South  (+1,  0)
    2 : move West   ( 0, -1)
    3 : move East   ( 0, +1)
    """

    actions = {
        0: (-1,  0),   # North
        1: ( 1,  0),   # South
        2: ( 0, -1),   # West
        3: ( 0,  1),   # East
    }

    def __init__(self, streets, dir_limit,
                 boundary=-10, against=-20, terminal=200):
        """
        Parameters
        ----------
        streets   : np.ndarray (y, x)  0 = open road, 1 = blocked
        dir_limit : np.ndarray (N, 4)  1 = direction allowed, 0 = forbidden
        boundary  : reward for hitting a boundary or blocked cell
        against   : reward for moving against a forbidden direction
        terminal  : reward for reaching the goal node
        """
        self.streets   = streets
        self.dir_limit = dir_limit
        self.boundary  = boundary
        self.against   = against
        self.terminal  = terminal

        self.y_range, self.x_range = self.streets.shape

        self.goal_pos  = None
        self.agent_pos = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def act_len(self):
        """Number of available actions."""
        return len(self.actions)

    def block_len(self):
        """Total number of nodes in the grid."""
        return self.y_range * self.x_range

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def get_idx(self, pos=None):
        """Convert (row, col) coordinates to a flat node index."""
        if pos is None:
            pos = self.agent_pos
        r, c = pos
        return self.x_range * r + c

    def get_coord(self, idx):
        """Convert a flat node index to (row, col) coordinates."""
        return (idx // self.x_range, idx % self.x_range)

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self, start_pos, end_loc_coords=None):
        """Initialise agent and goal positions for a new episode.

        Parameters
        ----------
        start_pos      : int  flat node index for the starting position
        end_loc_coords : int or None  flat node index for the goal;
                         defaults to the bottom-right corner

        Returns
        -------
        int  flat node index of the starting state
        """
        self.agent_pos = self.get_coord(start_pos)

        if end_loc_coords is not None:
            self.goal_pos = self.get_coord(end_loc_coords)
        else:
            self.goal_pos = self.get_coord(self.block_len() - 1)

        return self.get_idx()

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def penalty_dis(self, alpha):
        """Manhattan-distance penalty from current position to goal.

        Parameters
        ----------
        alpha : float  weight multiplied by the Manhattan distance

        Returns
        -------
        float  non-negative distance penalty
        """
        distance = (abs(self.goal_pos[0] - self.agent_pos[0]) +
                    abs(self.goal_pos[1] - self.agent_pos[1]))
        return alpha * distance

    def penalty_cong(self, beta, overflow, lst_trf, lstm_predict):
        """Multi-step congestion penalty using LSTM traffic predictions.

        Simulates the downstream queue evolution over the next
        ``len(lstm_predict)`` time steps and accumulates a discounted
        congestion cost.

        Parameters
        ----------
        beta         : float  discount factor for future steps
        overflow     : tuple  (mean_flow, variance) of the lane throughput
        lst_trf      : float  current queued vehicle count for this lane
        lstm_predict : array-like  predicted future flows (1-D, length T)

        Returns
        -------
        float  total congestion penalty
        """
        penalty   = 0
        ave_overflow, variance = overflow
        lst_traffic = lst_trf
        cost_val    = 1

        for i in range(len(lstm_predict) + 1):
            lane_reward = int(np.random.normal(ave_overflow, np.sqrt(variance)))
            lst_traffic = lstm_predict[i - 1] if i > 0 else lst_traffic
            lst_traffic -= lane_reward
            lst_traffic  = max(0, lst_traffic)

            cong_level = get_state_index(lst_traffic)
            penalty   += cong_level * (beta ** cost_val)
            cost_val  += 1

        return penalty

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, alpha, action, penalty):
        """Execute one action and return the transition tuple.

        Parameters
        ----------
        alpha   : float  distance-penalty weight
        action  : int    action index (0–3)
        penalty : float  pre-computed congestion penalty for this step

        Returns
        -------
        next_state : int    flat node index after the step
        reward     : float  immediate reward signal
        done       : bool   True if the goal was reached
        """
        dr, dc   = self.actions[action]
        r, c     = self.agent_pos
        new_r, new_c = r + dr, c + dc

        if (new_r < 0 or new_r >= self.y_range or
                new_c < 0 or new_c >= self.x_range or
                self.streets[new_r, new_c] == 1):
            # Out of bounds or blocked cell
            reward   = self.boundary
            done     = False
            next_pos = self.agent_pos

        elif self.dir_limit[self.get_idx(self.agent_pos), action] == 0:
            # Forbidden direction (one-way road violation)
            reward   = self.against
            done     = False
            next_pos = self.agent_pos

        else:
            next_pos = (new_r, new_c)
            if self.goal_pos is not None and next_pos == self.goal_pos:
                reward = self.terminal
                done   = True
            else:
                reward = -penalty
                done   = False

        self.agent_pos = next_pos
        if not done:
            reward -= self.penalty_dis(alpha)

        return self.get_idx(), reward, done
