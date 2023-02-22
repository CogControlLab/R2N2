import matplotlib
# matplotlib.use("Agg")

from copy import deepcopy
from time import sleep
from tqdm import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


class TMazeNavTask:
    """Agent is required to turn left/right according to the onset cue type.
    1 represents left reward while 2 represents right reward.
    """
    def __init__(self, l_cue, l_delay1, l_branch, cue_type=1):
        self.l_cue = l_cue
        self.l_delay1 = l_delay1
        self.l_branch = l_branch

        self.end_coords = ((self.l_cue + self.l_delay1, 0),
                           (self.l_cue + self.l_delay1, self.l_branch * 2))
        self.set_cue_type(cue_type)
        self.reset()

    def reset(self, ):
        self.current_coords = np.array([0, self.l_branch])
        return self.observe(self.current_coords)

    def set_cue_type(self, cue_type):
        self.cue_type = cue_type  # 1: reward left 2: reward right
        self.maze = np.zeros((self.l_cue + self.l_delay1 + 1,
                              self.l_branch * 2 + 1)) - 100  # wall
        self.maze[:, self.l_branch] = 0  # normal tube
        self.maze[-1, :] = 0  # normal tube
        self.maze[:self.l_cue, self.l_branch] = cue_type

        # also set other parts. delay period: 0; turning point: 3, branch: 4
        self.maze[-1, :] = 4
        self.maze[-1, self.l_branch] = 3

        self.rewards = np.zeros_like(self.maze)
        if self.cue_type == 1:
            self.rewards[-1, 0] = 1
        elif self.cue_type == 2:
            self.rewards[-1, -1] = 1

        s = 0
        self.states_mapping = {}
        for x in range(self.l_cue + self.l_delay1 + 1):
            for y in range(self.l_branch * 2 + 1):
                if self.maze[x, y] != -100:
                    s += 1
                    self.states_mapping[(x, y)] = s

    def get_state_n(self, coords):
        return self.states_mapping[coords]

    def get_manhattan_distance(self, coords):

        coords_start = (0, self.l_branch)
        return np.sum(
            [np.abs(xi - yi) for xi, yi in zip(coords_start, coords)])

        pass

    @staticmethod
    def get_next_pos(coords, action):
        # 0:Left 1:Up 2:Right 3: Down
        return np.array(coords) + np.array([[0, -1], [-1, 0], [0, 1], [1, 0]
                                            ][action])

    def observe(self, coords):
        return self.maze[tuple(coords)]

    def step(self, action):
        next_coords = self.get_next_pos(self.current_coords, action)

        if (next_coords[0] < 0 or next_coords[0] > self.l_delay1 + self.l_cue) or \
                (next_coords[1] < 0 or next_coords[1] > self.l_branch * 2):
            next_coords = self.current_coords

        if self.observe(next_coords) == -100:  # wall
            next_coords = self.current_coords

        self.current_coords = next_coords
        observation = self.observe(self.current_coords)
        reward = self.rewards[tuple(self.current_coords)] == 1.0

        done = True if tuple(self.current_coords) in self.end_coords else False
        return observation, reward, done


if __name__ == "__main__":
    cue_length = 3
    delay_length = 2
    branch_length = 5
    cue_type = 2
    task = TMazeNavTask(cue_length, delay_length, branch_length, cue_type)
    observation = task.reset()
    da = {0: "L", 1: "U", 2: "R", 3: "D"}

    cmap = colors.ListedColormap(
        ['black', 'green', 'blue', 'red', 'orange', 'green'])
    bounds = [-100, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(task.maze, cmap=cmap, norm=norm)
    r_coord = 0 if cue_type == 1 else task.l_branch * 2
    plt.scatter(r_coord, task.l_cue + task.l_delay1, c='w', marker='*', s=100)

    for t in range(1000):
        a = np.random.choice(4, p=[0.2, 0.2, 0.3, 0.3])
        prev_coords = tuple(task.current_coords)
        observation, reward, done = task.step(a)
        current_coords = tuple(task.current_coords)
        if prev_coords != current_coords:
            task.get_manhattan_distance(current_coords)
            plt.plot([prev_coords[1], current_coords[1]],
                     [prev_coords[0], current_coords[0]],
                     marker='o',
                     c='white')
        if done:
            print(
                f"T = {t+1}| action: {da[a]}| prev: {prev_coords}| new: {current_coords}| observation: {observation}| reward: {reward} | done: {done}"
            )
            break

    plt.show()
    # plt.savefig("test.png")
