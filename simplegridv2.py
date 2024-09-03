# Import necessary modules and classes
from __future__ import annotations  # Enable future annotations
from minigrid.core.constants import COLOR_NAMES  # Import color constants
from minigrid.core.grid import Grid  # Import Grid class for environment layout
from minigrid.core.mission import MissionSpace  # Import MissionSpace for defining missions
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Floor  # Import environment objects
from minigrid.manual_control import ManualControl  # Import manual control for testing
from minigrid.minigrid_env import MiniGridEnv  # Import base environment class

class SimpleEnv(MiniGridEnv):
    """
    A custom environment class that extends MiniGridEnv.
    This environment creates a simple grid world with walls, lava, keys (floor tiles), and a goal.
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        """
        Initialize the SimpleEnv.

        Args:
        size: Size of the square grid
        agent_start_pos: Starting position of the agent
        agent_start_dir: Starting direction of the agent
        max_steps: Maximum number of steps per episode
        **kwargs: Additional keyword arguments
        """
        self.agent_start_pos = agent_start_pos  # Set agent's starting position
        self.agent_start_dir = agent_start_dir  # Set agent's starting direction
        self.key_positions = []  # Initialize list to store key positions (will be set in _gen_grid)
        
        # Create a MissionSpace object with the _gen_mission method
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        # Set max_steps to 4 * size^2 if not provided
        if max_steps is None:
            max_steps = 4 * size**2
        
        # Call the parent class constructor with the initialized parameters
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        """
        Generate the mission statement for the environment.

        Returns:
        str: The mission statement
        """
        return "Collect all keys and reach the goal"

    def _gen_grid(self, width, height):
        """
        Generate the grid environment.

        Args:
        width: Width of the grid
        height: Height of the grid
        """
        self.grid = Grid(width, height)  # Create an empty grid
        self.grid.wall_rect(0, 0, width, height)  # Add walls around the perimeter
        
        # Place fixed walls at specific positions
        wall_positions = [(2, 2), (2, 3), (3, 2), (5, 5), (5, 6)]
        for x, y in wall_positions:
            self.put_obj(Wall(), x, y)
        
        # Place fixed lava at specific positions
        lava_positions = [(4, 4), (4, 5)]
        for x, y in lava_positions:
            self.put_obj(Lava(), x, y)
        
        # Place fixed floor tiles (blue keys) at specific positions
        self.key_positions = [(3, 5), (6, 3)]
        for x, y in self.key_positions:
            self.put_obj(Floor('blue'), x, y)
        
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        
        # Place the agent at the start position or randomly if not specified
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        
        # Set the mission
        self.mission = "Collect all keys and reach the goal"

    def reset(self, **kwargs):
        """
        Reset the environment to its initial state.

        Returns:
        The result of the parent class reset method
        """
        self.stepped_floors = set()  # Reset the set of stepped floors
        return super().reset(**kwargs)  # Call the parent class reset method

    def step(self, action):
        """
        Perform one step in the environment.

        Args:
        action: The action to take

        Returns:
        obs: The observation after taking the action
        reward: The reward received
        terminated: Whether the episode has terminated
        truncated: Whether the episode has been truncated
        info: Additional information
        """
        # Perform a step in the environment using the parent class method
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Check if the agent stepped on lava
        if isinstance(self.grid.get(*self.agent_pos), Lava):
            reward -= 1  # Negative reward for stepping on lava
            terminated = True
            print(f"Stepped on Lava! Negative reward: {reward}")
        
        # Check if the agent stepped on a floor tile (key)
        if isinstance(self.grid.get(*self.agent_pos), Floor):
            if self.agent_pos not in self.stepped_floors:
                reward += 0.1  # Positive reward for stepping on a new floor
                self.stepped_floors.add(self.agent_pos)
                print(f"Stepped on new Floor! Positive reward: {reward}")
            else:
                print("Stepped on already visited Floor. No additional reward.")
        
        # Check if the agent reached the goal
        if isinstance(self.grid.get(*self.agent_pos), Goal):
            if len(self.stepped_floors) == len(self.key_positions):
                reward += 1  # Additional reward for reaching the goal with all keys
                terminated = True
                print(f"Reached Goal with all keys! Reward: {reward}")
            else:
                print("Reached Goal, but not all keys collected.")
        
        return obs, reward, terminated, truncated, info

def main():
    """
    Main function to run the environment with manual control.
    """
    env = SimpleEnv(render_mode="human")  # Create an instance of SimpleEnv with human-readable rendering
    manual_control = ManualControl(env, seed=42)  # Enable manual control for testing
    manual_control.start()  # Start the manual control

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()