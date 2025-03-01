class Grid:
    def __init__(self, grid_size, start_state, goal_state, obstacles, reward, cost=-1):
        """
        Initialize the Grid-World environment.

        Args:
            grid_size (int):            Size of the grid (grid_size x grid_size).
            start_state (tuple):        Starting position of the agent (x, y).
            goal_state (tuple):         Goal position (x, y).
            obstacles (list of tuples): List of obstacle positions [(x1, y1), (x2, y2), ...].
            reward (float):             Reward for reaching the goal.
            cost (float):               Cost for each step (default: -1).
        """
        self.grid_size   = grid_size
        self.start_state = start_state
        self.goal_state  = goal_state
        self.obstacles   = obstacles
        self.actions     = [0, 1, 2, 3, 4]  # Stay, Up, Down, Left, Right
        self.num_actions = len(self.actions)
        self.goal_reward = reward
        self.cost        = cost
        

    def simulate_action(self, state, action):
        """
        Simulate an action and return the new state.

        Args:
            state (tuple): Current position (x, y).
            action (int): Action to take (0: Up, 1: Down, 2: Left, 3: Right).

        Returns:
            tuple: New position (x, y) after taking the action.
        """
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}. Valid actions are {self.actions}")
        x_prev, y_prev = state  # Current position
        if action == 0:   # Stay
            x, y = x_prev, y_prev
        elif action == 1:  # Up
            x = max(x_prev - 1, 0)
            y = y_prev
        elif action == 2:  # Down
            x = min(x_prev + 1, self.grid_size - 1)
            y = y_prev
        elif action == 3:  # Left
            x = x_prev
            y = max(y_prev - 1, 0)
        elif action == 4:  # Right
            x = x_prev
            y = min(y_prev + 1, self.grid_size - 1)

        # Check if the new state is valid
        if not self._is_valid_state((x, y)):
            return (x_prev, y_prev)

        return (x, y)

    def _is_valid_state(self, state):
        """
        Check if a state is valid (within the grid and not an obstacle).

        Args:
            state (tuple): Position (x, y) to check.

        Returns:
            bool: True if the state is valid, False otherwise.
        """
        x, y = state
        return (0 <= x < self.grid_size and   # Check grid boundaries
                0 <= y < self.grid_size and   # Check grid boundaries
                state not in self.obstacles)  # Check for obstacles

    def is_terminal(self, state):
        """
        Check if a state is terminal (goal or obstacle).

        Args:
            state (tuple): Position (x, y) to check.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        return state == self.goal_state or state in self.obstacles

    def get_reward(self, state):
        """
        Get the reward for a state.

        Args:
            state (tuple): Position (x, y) to evaluate.

        Returns:
            float: Reward for the state.
        """
        if not len(self.goal_state):
            return self.goal_reward
        elif state == self.goal_state:
            return self.goal_reward  # Reward for reaching the goal
        else:
            return self.cost         # Cost for each step