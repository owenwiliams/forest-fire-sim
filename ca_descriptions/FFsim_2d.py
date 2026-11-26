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

# --- CONFIGURATION VARIABLES ---

map_path = "town_map.txt"
IGNITION = "I" # P for power plant, I for incinerator start

WIND_FROM = "N" # direction wind is coming FROM
WIND_SPEED = 8 # speed in m/s
WIND_SPEED_FACTOR, WIND_DIRECTION_FACTOR = 0.045, 0.131

GENERATIONS = 1000
HOURS_PER_GEN = 1 # 24 generations = roughly one day

# water budget is 12.5km^2 so 8 tiles
MAX_WATER_TILES = 8
water_tiles = [(35, 11), (35, 12), (35, 13), (36, 11), (36, 13), (37, 11), (37, 12), (37, 13)]

# --- END ---

class States(IntEnum):
    Burnt = 0
    Burning = 1

    Canyon = 2
    Chaparral = 3
    Forest = 4

    Lake = 5
    Town = 6
    Hazard = 7 # power plant or incinerator
    Wet = 8

# this variable basically is the base probability of a square setting on fire per burning
# neighbour. eg. if probability is 0.5, having 1 burning neighbour gives it 50% chance of burning,
# and 2 burning neighbours brings it to 75%
BURN_PROBABILITIES = {
    States.Canyon: 0.1,
    States.Chaparral: 0.02,
    States.Forest: 0.004,
    States.Town: 0.12
}

# this is how long a burning square will continue to burn for (in generations)
# first figure is minimum second is mamximum (we'll use it as a range)

# rn im assuming 10 is about a day
BURN_DURATION = {
    States.Canyon: (6,8),
    States.Chaparral: (12,72), 
    States.Forest: (120, 720), 
    States.Town: (36,48) 
}
WET_DURATION = (4,8)

timers = None
old_tile = None

HAZARD_POS = {} # filled in by load_map
initial_terrain = None

# turns compass directions into a row and column step on the grid
COMPASS = {"N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
           "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1)}
# --- DO NOT CHANGE THE ORDER, HAS TO BE THIS WAY ---
# due to a weird capyle thing, otherwise weights get attached to the wrong neighbours
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

#
def wind_weights():
    # flips wind dir for direction wind is blowing towards
    wind_to = -np.array(COMPASS[WIND_FROM], float)
    wind_to /= np.linalg.norm(wind_to) # so wind diagonals are counted as strong
    weights = []
    # goes through neighbours one at a time, checking fire spread diagonal weights
    for dr, dc in OFFSETS:
        spread = -np.array([dr, dc], float)
        spread /= np.linalg.norm(spread)
        cos_theta = spread @ wind_to
        # scores how well the fire's direction lines up with the wind
        weights.append(np.exp(WIND_SPEED * (WIND_SPEED_FACTOR + WIND_DIRECTION_FACTOR * (cos_theta - 1))))
    return weights

# runs function once and saves numbers (only needs to run at start)
WEIGHTS = wind_weights()
print("Wind weights (NW, N, NE, W, E, SW, S, SE):", [round(float(w), 2) for w in WEIGHTS])

def setup(args):
    """Set up the config object used to interact with the GUI"""
    config_path = args[0]
    config = utils.load(config_path)
    # -- THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED --
    config.title = "Forest fire sim"
    config.dimensions = 2
    config.states = tuple(s.value for s in States)
    config.wrap = False
    config.num_generations = GENERATIONS

    global timers
    global old_tile

    timers = None
    old_tile = {}

    config.state_colors = [
        (66/255, 44/255, 35/255),    # 0: Burnt
        (255/255, 98/255, 31/255),   # 1: Burning
        (255/255, 253/255, 75/255),  # 2: Canyon
        (191/255, 189/255, 55/255),  # 3: Chaparral
        (77/255, 98/255, 47/255),    # 4: Forest
        (0, 181/255, 235/255),   # 5: Lake
        (0, 0, 0),       # 6: Town
        (0, 0, 0),       # 7: Hazard
        (150/255, 220/255, 255/255), # 8: Wet
    ]

    # loads the map, saves a copy of the terrain. first frame seen is already burning
    config.grid_dims = (40,40)
    grid = load_map("maps/" + map_path)
    global initial_terrain
    initial_terrain = grid.copy()
    if len(set(water_tiles)) > MAX_WATER_TILES:
        raise ValueError(f"Too many water tiles: {len(set(water_tiles))} given, max is {MAX_WATER_TILES}")
    grid = ignite_at_hazard(grid, IGNITION)
    config.initial_grid = grid

    if len(args) == 2:
        config.save()
        sys.exit()
    return config


def transition_function(grid, neighbourstates, neighbourcounts):
    """Function to apply the transition rules
    and return the new grid"""

    # set up timers grid if not set already
    global timers
    global old_tile

    if timers is None or timers.shape != grid.shape:
        timers = np.zeros(grid.shape, dtype=int)
        # giving first cells a burn time
        for r,c in zip(*np.where(grid == States.Burning)):
            lb,ub = BURN_DURATION[States(int(initial_terrain[r,c]))]
            timers[r,c] = np.random.randint(lb,ub+1)

    # incorporate wind changing prob, would mean changing how "burning_neighbours" works
    # (for example will weigh a burning neighbour on windy side higher than one on other side)
    NW, N, NE, W, E, SW, S, SE = neighbourstates
    all_neighbours = [NW, N, NE, W, E, SW, S, SE]

    burning_count = sum((nb == States.Burning) for nb in all_neighbours)
    # hazard_count = sum((nb == States.Hazard) for nb in all_neighbours)
    lake_count = sum((nb == States.Lake) for nb in all_neighbours)

    burning_neighbours = np.maximum(
        burning_count - (0.2 * lake_count),
        0
    )

    new_grid = grid.copy()
    rows, cols = grid.shape

    # loop the grid, applying rules based on its state rn
    for row in range(rows):
        for col in range(cols):
            bn = burning_neighbours[row, col]
            state = grid[row, col]

            match state:
                # these ones wont change or burn
                case States.Lake | States.Hazard | States.Burnt:
                    pass

                # decrease the burn timer, when depleted go to burnt                
                case States.Burning:
                    timers[row, col] -= 1
                    if timers[row, col] <= 0:
                        new_grid[row, col] = States.Burnt

                # if the tile is a wet tile it is invincible for a short time
                case States.Wet:
                    timers[row, col] -= 1
                    if timers[row, col] <= 0:
                        new_grid[row, col] = old_tile[(row, col)]                  

                # the rest all work same, try setting on fire based on neighbours and their
                # state's probability of burning
                case _:
                    if try_ignite(state, bn):
                        # if water dropped here, it goes wet instead of catching fire
                        if (row, col) in water_tiles:
                            new_grid[row, col] = States.Wet
                            timers[row,col] = WET_DURATION
                            old_tile[(row, col)] = state
                            water_tiles.remove((row,col))
                            continue
                        new_grid[row, col] = States.Burning
                        (lb, ub) = BURN_DURATION[state]
                        # sets on fire for random duration in range
                        timers[row, col] = np.random.randint(lb, ub + 1)

    grid[:] = new_grid
    return grid

# to make wind direction work we'll have to check neighbours, give them a value based on
# wind and pass that instead as "burning neighbours"
# eg. counter wind will be 0.5x or something, and with wind 2x maybe
def try_ignite(state, burning_neighbours):
    if burning_neighbours <= 0:
        return False
    p = BURN_PROBABILITIES[state]
    cumulative_prob = 1.0 - (1.0 - p) ** burning_neighbours
    return np.random.random() < cumulative_prob

# looks at the 8 cells around chosen hazard and sets any fuel cells to burning
def ignite_at_hazard(grid, which):
    r0,c0 = HAZARD_POS[which]
    rows,cols = grid.shape
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            r,c = r0 + dr, c0 +dc
            if 0 <= r < rows and 0 <= c < cols and grid[r,c] in (States.Canyon,States.Chaparral, States.Forest):
                # water tiles go wet instead of starting to burn
                grid[r,c] = States.Wet if (r, c) in water_tiles else States.Burning
    return grid

def load_map(path):
    with open(path, "r") as f:
        rows = [line.strip().split() for line in f if line.strip()]

    mapping = {
        "P": States.Hazard, # changed to accomodate both hazards, so we can pick which one to start fire at
        "I": States.Hazard,
        "L": States.Lake,
        "C": States.Canyon,
        "G": States.Chaparral,
        "F": States.Forest,
        "T": States.Town,
    }

    HAZARD_POS.clear()
    for r, row in enumerate(rows):
        for c,cell in enumerate(row):
            if cell in ("P", "I"):
                HAZARD_POS[cell] = (r,c)

    return np.array([
        [mapping[cell] for cell in row]
        for row in rows
    ], dtype=int)

def time_to_town(timeline):
    town = (initial_terrain == States.Town) # makes a true/false grid marking town cells
    for gen, g in enumerate(timeline):
        if (g[town] != States.Town).any():
            return gen # returns current generation if town is burnt
    return None
    

def main():
    """ Main function that sets up, runs and saves CA"""
    config = setup(sys.argv[1:])

    grid = Grid2D(config, transition_function)

    timeline = grid.run()
    gen = time_to_town(timeline)
    if IGNITION == "I":
        hazard = "Incinerator"
    elif IGNITION == "P":
        hazard = "Power plant"
    else:
        hazard = "N/A"
    if gen is None:
        print(f"Fire did not reach the town (fire started at {hazard}, wind from {WIND_FROM})")
    else:
        print(f"Fire reached the town at generation {gen} "
              f"(~{gen * HOURS_PER_GEN:.0f} hours) (fire started at {hazard}, wind from {WIND_FROM})")

    config.save()

    utils.save(timeline, config.timeline_path)

if __name__ == "__main__":
    main()
