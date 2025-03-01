# Importing libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from   matplotlib.animation import FuncAnimation
# Importing custom libraries
from   grid import Grid

# Function to visualize the learned policy
def visualize_policy(Q, grid):
    policy = np.zeros((grid.grid_size, grid.grid_size), dtype=str)
    # Iterating over the grid
    for x in range(grid.grid_size):
        for y in range(grid.grid_size):
            if (x, y) == grid.goal_state:
                policy[x, y] = "G"  # Goal
            elif (x, y) in grid.obstacles:
                policy[x, y] = "X"  # Obstacle
            else:
                policy[x, y] = ["stay", "up", "down", "left", "right"][np.argmax(Q[x, y])]  # Best action
    print("Learned Policy:")
    print(policy)

# Function to visualize the agent's path through the grid
def visualize_agent_path(grid, path):
    # Create a grid representation
    fig, ax = plt.subplots(figsize=(grid.grid_size, grid.grid_size))
    ax.set_xlim(0, grid.grid_size)
    ax.set_ylim(0, grid.grid_size)
    ax.set_xticks(np.arange(0, grid.grid_size + 1))
    ax.set_yticks(np.arange(0, grid.grid_size + 1))
    ax.grid(True)
    # Remove axis labels (but keep ticks and grid)
    plt.gca().set_xticklabels([]); plt.gca().set_yticklabels([])
    # Flip the y-axis
    ax.invert_yaxis()
    # Plot obstacles
    for obstacle in grid.obstacles:
        ax.add_patch(plt.Rectangle((obstacle[1], obstacle[0]), 1, 1, color='red', alpha=0.5))
    # Plot start state
    ax.add_patch(plt.Rectangle((grid.start_state[1], grid.start_state[0]), 1, 1, color='blue', alpha=0.5, label='Start'))
    # Plot goal state
    if len(grid.goal_state) > 0:
        ax.add_patch(plt.Rectangle((grid.goal_state[1], grid.goal_state[0]), 1, 1, color='green', alpha=0.5, label='Goal'))
    # Plot the agent's path with arrows
    for i in range(len(path) - 1):
        current_state = path[i]
        next_state    = path[i + 1]
        # Calculate the center of the current and next states
        current_x = current_state[1] + 0.5
        current_y = current_state[0] + 0.5  # No inversion for y-axis
        next_x    = next_state[1] + 0.5
        next_y    = next_state[0] + 0.5  # No inversion for y-axis
        # Draw an arrow from the current state to the next state
        ax.arrow(
            current_x, current_y,  # Start point
            next_x - current_x, next_y - current_y,  # Direction
            head_width=0.2, head_length=0.2, fc='purple', ec='black'
        )
    # Add labels and legend
    ax.set_title("Agent's Path (Post Training)", fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.show()

# Function to get the agent's path
def test_agent(Q, grid, max_steps_per_episode=100):
    state         = grid.start_state
    path          = [state]
    episode_steps = 0
    # Simulate the agent's actions until it reaches a goal-state
    while not grid.is_terminal(state):
        action = np.argmax(Q[state[0], state[1]])
        state  = grid.simulate_action(state, action)
        path.append(state)
        episode_steps += 1
        # If the agent exceeds maximum steps
        if episode_steps >= max_steps_per_episode:
            break
    print("Agent's Path:")
    print(path)

# Q-learning training loop
def train_agent(Q_init, grid, alpha, gamma, epsilon, num_episodes=1000, max_steps_per_episode=1000):
    # Initialize Q-table
    Q = Q_init.copy()
    # Initialize state-visit counter
    state_visits = np.zeros((grid.grid_size, grid.grid_size), dtype=float)
    # Iterating over the episodes
    for episode in range(num_episodes):
        state = grid.start_state
        steps = 0
        # Simulating an episode until the agent reaches a terminal state
        while not grid.is_terminal(state) and steps < max_steps_per_episode:
            # Increment the visitation count for the current state
            state_visits[state[0], state[1]] += 1
            # Choose action: epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action = np.random.choice(grid.num_actions)  # Explore
            else:
                action = np.argmax(Q[state[0], state[1]])  # Exploit
            # Simulate action and observe next state and reward
            next_state = grid.simulate_action(state, action)
            # Stay actions should incur no cost
            reward = grid.get_reward(next_state) * np.sign(action)
            # Update Q-value using the Q-learning formula
            Q[state[0], state[1], action] += alpha * (
                reward + gamma * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action]
            )
            # Move to the next state
            state = next_state
            steps += 1
        # Print progress (optional)
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed.")
    state_visits /= np.max(state_visits)  # Normalize state visitation counts
    return Q, state_visits

# Function to evaluate the agent's performance
def evaluate_agent(Q, grid, random = False, num_test_episodes=100, max_steps_per_episode=100):
    success_count = 0
    total_steps   = 0
    total_reward  = []
    # Initializing a state-visit counter
    state_visits = np.zeros((grid.grid_size, grid.grid_size), dtype=float)
    # Iterating over the test episodes
    for episode in range(num_test_episodes):
        state          = grid.start_state
        episode_steps  = 0
        episode_reward = 0
        # Simulating an episode until the agent reaches a terminal state
        while not grid.is_terminal(state):
            # Increment the visitation count for the current state
            state_visits[state[0], state[1]] += 1
            if not random:
                action = np.argmax(Q[state[0], state[1]])  # Greedy policy
            else:
                action = np.random.choice(grid.num_actions)
            next_state = grid.simulate_action(state, action)
            # Stay actions should incur no cost
            reward = grid.get_reward(next_state) * np.sign(action)
            episode_reward += reward
            state = next_state
            episode_steps += 1
            # If the agent exceeds maximum steps
            if episode_steps >= max_steps_per_episode:
                break
        if state == grid.goal_state:
            success_count += 1
        total_steps += episode_steps
        total_reward.append(episode_reward)
    # Calculate metrics
    success_rate  = success_count / num_test_episodes
    avg_steps     = total_steps   / num_test_episodes
    avg_reward    = np.mean(total_reward)
    state_visits /= np.max(state_visits)  # Normalize state visitation counts
    # Printing the results
    print("\nEvaluation Results:")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Average Steps to Goal: {avg_steps:.2f}")
    print(f"Average Reward per Episode: {avg_reward:.2f}")

    return success_rate, avg_steps, avg_reward



if __name__ == '__main__':
    # Creating the grid
    grid_size   = 6  # 5x5 grid
    start_state = (0, 0)  # Starting position
    goal_state  = (5, 5)  # Goal position
    obstacles   = []  # Obstacle positions

    # Reward and punishment
    goal_reward = 25
    cost        = -1

    # Environment
    grid = Grid(grid_size, start_state, goal_state, obstacles, goal_reward, cost)

    # Q-learning parameters
    alpha                 = 0.256  # Learning rate
    gamma                 = 0.9    # Discount factor
    epsilon               = 0.25   # Exploration rate
    num_episodes          = 10000  # Number of training episode
    max_steps_per_episode = 10000  # Maximum steps per episode

    # Training the agent
    Q_init = np.zeros((grid.grid_size, grid.grid_size, grid.num_actions))
    Q, _      = train_agent(Q_init, grid, alpha, gamma, epsilon, num_episodes, max_steps_per_episode)

    # Test the agent and get its path
    state = grid.start_state
    path  = [state]
    while not grid.is_terminal(state):
        action = np.argmax(Q[state[0], state[1]])
        state = grid.simulate_action(state, action)
        path.append(state)

    # Visualize the agent's path
    # visualize_agent_path(grid, path)

    # Visualize the learned policy and test the agent
    visualize_policy(Q, grid)
    test_agent(Q, grid)
    visualize_agent_path(grid, path)

    # Evaluate the agent's performance
    evaluate_agent(Q, grid, num_test_episodes=100)

    max_steps_per_episode = 1000
    num_episodes          = 100

    # Outcome devaluation grid (No Reward)
    print('\nOutcome Devaluation (No Reward):')
    devalued_grid = Grid(grid_size, start_state, goal_state, obstacles, (goal_reward * 0), cost)
    # Training the goal-directed agent
    Q_devalued, _       = train_agent(Q, devalued_grid, alpha, gamma, epsilon, num_episodes, max_steps_per_episode)
    evaluate_agent(Q_devalued, devalued_grid, num_test_episodes=100)
    visualize_policy(Q_devalued, devalued_grid)
    # Training the habit agent
    Q_devalued_habit, _ = train_agent(Q, devalued_grid, (alpha * 0), gamma, epsilon, num_episodes, max_steps_per_episode)
    evaluate_agent(Q_devalued_habit, devalued_grid, num_test_episodes=100)
    visualize_policy(Q_devalued_habit, devalued_grid)


    # Outcome devaluation (Negative Reward)
    print('\nOutcome Devaluation (Negative Reward):')
    devalued_grid = Grid(grid_size, start_state, goal_state, obstacles, (goal_reward * -1.), cost)
    # Training the goal-directed agent
    Q_devalued, _       = train_agent(Q, devalued_grid, alpha, gamma, epsilon, num_episodes, max_steps_per_episode)
    evaluate_agent(Q_devalued, devalued_grid, num_test_episodes=100)
    visualize_policy(Q_devalued, devalued_grid)
    # Training the habit agent
    Q_devalued_habit, _ = train_agent(Q, devalued_grid, (alpha * 0), gamma, epsilon, num_episodes, max_steps_per_episode)
    evaluate_agent(Q_devalued_habit, devalued_grid, num_test_episodes=100)
    visualize_policy(Q_devalued_habit, devalued_grid)


    # Contingency degradation
    print('\nContingency Degradation:')
    contingency_degraded_grid = Grid(grid_size, start_state, (), obstacles, goal_reward, cost)
    # Training the goal-directed agent
    Q_contingency_degraded, state_visits       = train_agent(Q, contingency_degraded_grid, alpha, gamma, epsilon, num_episodes, max_steps_per_episode)
    evaluate_agent(Q_contingency_degraded, contingency_degraded_grid, num_test_episodes=100)
    visualize_policy(Q_contingency_degraded, contingency_degraded_grid)
    # Training the habit agent
    Q_contingency_degraded_habit, state_visits = train_agent(Q, contingency_degraded_grid, (alpha * 0), gamma, epsilon, num_episodes, max_steps_per_episode)
    evaluate_agent(Q_contingency_degraded_habit, contingency_degraded_grid, num_test_episodes=100)
    visualize_policy(Q_contingency_degraded_habit, contingency_degraded_grid)


    # Novel Goal (Change in Goal Location)
    print('\nNovel Goal:')
    novel_grid = Grid(grid_size, start_state, (0, 5), obstacles, goal_reward, cost)
    # Training the goal-directed agent
    Q_novel, _       = train_agent(Q, novel_grid, alpha, gamma, epsilon, num_episodes, max_steps_per_episode)
    evaluate_agent(Q_novel, novel_grid, num_test_episodes=100)
    visualize_policy(Q_novel, novel_grid)
    # Training the habit agent
    Q_novel_habit, _ = train_agent(Q, novel_grid, (alpha * 0), gamma, epsilon, num_episodes, max_steps_per_episode)
    evaluate_agent(Q_novel_habit, novel_grid, num_test_episodes=100)
    visualize_policy(Q_novel_habit, novel_grid)

    alpha_range = [0.0, 0.004, 0.016, 0.032, 0.064, 0.256]
    for alpha in alpha_range:
        print(f'\nAlpha: {alpha}')
        Q_novel, _ = train_agent(Q, novel_grid, alpha, gamma, epsilon, num_episodes, max_steps_per_episode)
        success_rate, avg_steps, avg_reward = evaluate_agent(Q_novel, novel_grid, num_test_episodes=100)
        visualize_policy(Q_novel, novel_grid)



    # Novel Goal (Obstacle)
    novel_grid = Grid(grid_size, start_state, goal_state, [(5, 4)], goal_reward, cost)
    # Training the goal-directed agent
    Q_novel, _       = train_agent(Q, novel_grid, alpha, gamma, epsilon, num_episodes, max_steps_per_episode)
    evaluate_agent(Q_novel, novel_grid, num_test_episodes=100)
    visualize_policy(Q_novel, novel_grid)
    # Training the habit agent
    Q_novel_habit, _ = train_agent(Q, novel_grid, (alpha * 0), gamma, epsilon, num_episodes, max_steps_per_episode)
    evaluate_agent(Q_novel_habit, novel_grid, num_test_episodes=100)
    visualize_policy(Q_novel_habit, novel_grid)
    
    average_rewards = []

    alpha_range = [0.0, 0.00004, 0.00008, 0.00016]
    for alpha in alpha_range:
        print(f'\nAlpha: {alpha}')
        Q_novel, _ = train_agent(Q, novel_grid, alpha, gamma, epsilon, num_episodes, max_steps_per_episode)
        success_rate, avg_steps, avg_reward = evaluate_agent(Q_novel, novel_grid, num_test_episodes=100)
        visualize_policy(Q_novel, novel_grid)
        average_rewards.append(avg_reward)

    # Random agent
    print('\nRandom Agent:')
    success_rate, avg_steps, avg_reward = evaluate_agent(Q, novel_grid, random=True, num_test_episodes=100)
    average_rewards = [avg_reward] + average_rewards
    print(average_rewards)