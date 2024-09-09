from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Floor
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=15,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.key_positions = []
        
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
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # Place walls in straight lines
        # Vertical walls
        for y in range(1, height-1):
            self.put_obj(Wall(), 3, y)
            self.put_obj(Wall(), 7, y)
            self.put_obj(Wall(), 11, y)
        
        # Horizontal walls
        for x in range(1, width-1):
            self.put_obj(Wall(), x, 3)
            self.put_obj(Wall(), x, 7)
        
        # Create openings in the walls
        openings = [(3, 2), (3, 6), (3, 10), 
                    (7, 4), (7, 8), (7, 12),
                    (11, 2), (11, 6), (11, 10),
                    (2, 3), (6, 3), (10, 3),
                    (4, 7), (8, 7), (12, 7),
                    (2, 11), (6, 11), (10, 11)]
        
        for x, y in openings:
            self.grid.set(x, y, None)
        
        # Place 5 fixed lava positions
        lava_positions = [(4, 1), (6, 6), (10, 11), (4, 12), (12, 4)]
        for x, y in lava_positions:
            self.put_obj(Lava(), x, y)
        
        # Place 5 fixed floor tiles (blue keys)
        self.key_positions = [(1, 5), (5, 9), (9, 13), (13, 1), (9, 5)]
        for x, y in self.key_positions:
            self.put_obj(Floor('blue'), x, y)
        
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        
        self.mission = "Collect all keys and reach the goal"

    def reset(self, **kwargs):
        self.stepped_floors = set()
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        if isinstance(self.grid.get(*self.agent_pos), Lava):
            reward -= 1
            terminated = True
            print(f"Stepped on Lava! Negative reward: {reward}")
        
        if isinstance(self.grid.get(*self.agent_pos), Floor):
            if self.agent_pos not in self.stepped_floors:
                reward += 0.1
                self.stepped_floors.add(self.agent_pos)
                print(f"Stepped on new Floor! Positive reward: {reward}")
            else:
                print("Stepped on already visited Floor. No additional reward.")
        
        if isinstance(self.grid.get(*self.agent_pos), Goal):
            if len(self.stepped_floors) == len(self.key_positions):
                reward += 1
                terminated = True
                print(f"Reached Goal with all keys! Reward: {reward}")
            else:
                print("Reached Goal, but not all keys collected.")
        
        return obs, reward, terminated, truncated, info

def main():
    env = SimpleEnv(render_mode="human")
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

if __name__ == "__main__":
    main()