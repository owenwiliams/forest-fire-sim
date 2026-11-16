# Name: Forest fire simulator
# Dimensions: 2

# --- Set up executable path, do not edit ---
import sys
import inspect
this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index('ca_descriptions')]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + 'capyle')
sys.path.append(main_dir_loc + 'capyle/ca')
sys.path.append(main_dir_loc + 'capyle/guicomponents')
# ---

from capyle.ca import Grid2D, Neighbourhood, randomise2d
import capyle.utils as utils
import numpy as np
from enum import IntEnum

class States(IntEnum):
    Burnt = 0
    Burning = 1

    Canyon = 2
    Chaparral = 3
    Forest = 4

    Lake = 5
    Town = 6
    Hazard = 7 # power plant or incinerator

# this variable basically is the base probability of a square setting on fire per burning
# neighbour. eg. if probability is 0.5, having 1 burning neighbour gives it 50% chance of burning,
# and 2 burning neighbours brings it to 75%
BURN_PROBABILITIES = {
    States.Canyon: 0.5,
    States.Chaparral: 0.4,
    States.Forest: 0.3,
    States.Town: 0.4, # town same as chaparral?
}

# this is how long a burning square will continue to burn for (in generations)
# first figure is minimum second is mamximum (we'll use it as a range)
BURN_DURATION = {
    States.Canyon: (4,5), # between 4 and 5 gens for canyon
    States.Chaparral: (15,20), # between 15 and 20 for chaparral
    States.Forest: (40, 60), # between 40 and 60 for dense forest
    States.Town: (100,100) # idk when towns on fire it stays on fire
}

burn_timers = None

class WindDir(IntEnum):
    N = 0
    E = 1
    S = 2
    W = 3 # do we want diagonals?

def setup(args):
    """Set up the config object used to interact with the GUI"""
    config_path = args[0]
    config = utils.load(config_path)
    # -- THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED --
    config.title = "Forest fire sim"
    config.dimensions = 2
    config.states = tuple(s.value for s in States)

    global burn_timers

    burn_timers = None

    config.state_colors = [
        (66, 44, 35),    # 0: Burnt
        (255, 98, 31),   # 1: Burning
        (255, 253, 75),  # 2: Canyon
        (191, 189, 55),  # 3: Chaparral
        (77, 98, 47),    # 4: Forest
        (0, 181, 235),   # 5: Lake
        (0, 0, 0),       # 6: Town
        (0, 0, 0),       # 7: Hazard
    ]

    config.grid_dims = (20,20)
    config.initial_grid = load_map("town_map.txt")

    if len(args) == 2:
        config.save()
        sys.exit()
    return config


def transition_function(grid, neighbourstates, neighbourcounts):
    """Function to apply the transition rules
    and return the new grid"""

    # set up timers grid if not set already
    global burn_timers
    if burn_timers is None or burn_timers.shape != grid.shape:
        burn_timers = np.zeros(grid.shape, dtype=int)

    # incorporate wind changing prob, would mean changing how "burning_neighbours" works
    # (for example will weigh a burning neighbour on windy side higher than one on other side)
    NW, N, NE, W, E, SW, S, SE = neighbourstates

    # ive made hazard count as a burning neighbour with lower value 
    # and lake neighbours removes a little too (harder to burn next to lake)
    burning_neighbors = neighbourcounts[States.Burning] + (0.2 * neighbourcounts[States.Hazard]) - (0.2 * neighbourcounts[States.Lake])
    # make sure not negative
    burning_neighbors = max(0.0, burning_neighbors)
    
    new_grid = grid.copy()
    rows, cols = grid.shape

    # loop the grid, applying rules based on its state rn
    for row in range(rows):
        for col in range(cols):
            bn = burning_neighbors[row, col]
            state = grid[row, col]

            match state:
                # these ones wont change or burn
                case States.Lake | States.Burnt | States.Hazard:
                    pass

                # decrease the burn timer, when depleted go to burnt                
                case States.Burning:
                    burn_timers[row, col] -= 1
                    if burn_timers[row, col] <= 0:
                        new_grid[row, col] = States.Burnt

                # the rest all work same, try setting on fire based on neighbours and their
                # state's probability of burning
                case _:
                    if try_ignite(state, bn):
                        new_grid[row, col] = States.Burning
                        (lb, ub) = BURN_DURATION[state]
                        # sets on fire for random duration in range
                        burn_timers[row, col] = np.random.randint(lb, ub + 1)


    
    return new_grid

# to make wind direction work we'll have to check neighbours, give them a value based on
# wind and pass that instead as "burning neighbours"
# eg. counter wind will be 0.5x or something, and with wind 2x maybe
def try_ignite(state, burning_neighbours):
    if burning_neighbours <= 0:
        return False
    p = BURN_PROBABILITIES[state]
    cumulative_prob = 1.0 - (1.0 - p) ** burning_neighbours
    return np.random.random() < cumulative_prob

def load_map(path):
    with open(path, "r") as f:
        rows = [line.strip().split() for line in f if line.strip()]

    mapping = {
        "H": States.Hazard,
        "L": States.Lake,
        "C": States.Canyon,
        "G": States.Chaparral,
        "F": States.Forest,
        "T": States.Town,
    }

    return np.array([
        [mapping[cell] for cell in row]
        for row in rows
    ], dtype=int)
    

def main():
    """ Main function that sets up, runs and saves CA"""
    config = setup(sys.argv[1:])

    grid = Grid2D(config, transition_function)

    timeline = grid.run()

    config.save()

    utils.save(timeline, config.timeline_path)

if __name__ == "__main__":
    main()
