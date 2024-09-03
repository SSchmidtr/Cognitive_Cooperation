from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Floor
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from random import randint

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.keys_collected = set()  # Track collected keys
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Collect all keys and reach the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Function to get random coordinates within the inner grid
        def get_random_inner_coords():
            return (randint(3, width-2), randint(3, height-2))

        # Function to place an object without overlapping
        def place_obj_no_overlap(obj_type, color=None):
            while True:
                x, y = get_random_inner_coords()
                if self.grid.get(x, y) is None:
                    if color:
                        self.grid.set(x, y, obj_type(color))
                    else:
                        self.grid.set(x, y, obj_type())
                    return x, y  # Return the coordinates where the object was placed

        # Place random walls
        for _ in range(5):
            place_obj_no_overlap(Wall)

        # Place random lava
        for _ in range(2):
            place_obj_no_overlap(Lava)

        # Place two keys
        self.key_positions = []
        for _ in range(2):
            x, y = place_obj_no_overlap(Floor, 'blue')
            self.key_positions.append((x, y))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Collect all keys and reach the goal"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        if isinstance(self.grid.get(*self.agent_pos), Lava):
            reward -= 1  # Negative reward for stepping on lava
            terminated = True
            print(f"Stepped on Lava! Negative reward: {reward}")

        # Check if a key was picked up

        if isinstance(self.grid.get(*self.agent_pos), Floor):
            reward += 0.1
            terminated = False
            print(f"Stepped on Floor! Positive reward: {reward}")

        # Check for goal
        if isinstance(self.grid.get(*self.agent_pos), Goal):
            if len(self.keys_collected) == len(self.key_positions):
                reward += 1  # Additional reward for reaching the goal with all keys
                terminated = True
                print(f"Reached Goal with all keys! Reward: {reward}")
            else:
                print("Reached Goal, but not all keys collected.")

        return obs, reward, terminated, truncated, info

def main():
    env = SimpleEnv(render_mode="human")
    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

if __name__ == "__main__":
    main()