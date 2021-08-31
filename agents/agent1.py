# https://www.kaggle.com/ilialar/lux-ai-risk-averse-baseline
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import math
import numpy as np

### Define helper functions

# this snippet finds all resources stored on the map and puts them into a list so we can search over them
def find_resources(game_state):
    resource_tiles: list[Cell] = []
    width, height = game_state.map_width, game_state.map_height
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)
    return resource_tiles

# the next snippet finds the closest resources that we can mine given position on a map
def find_closest_resources(pos, player, resource_tiles):
    closest_dist = math.inf
    closest_resource_tile = None
    for resource_tile in resource_tiles:
        # we skip over resources that we can't mine due to not having researched them
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = resource_tile.pos.distance_to(pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    return closest_resource_tile, closest_dist

def find_closest_city_tile(pos, player):
    closest_city_tile = None
    closest_dist = math.inf
    if len(player.cities) > 0:
        # the cities are stored as a dictionary mapping city id to the city object, which has a citytiles field that
        # contains the information of all citytiles in that city
        for k, city in player.cities.items():
            for city_tile in city.citytiles:
                dist = city_tile.pos.distance_to(pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_city_tile = city_tile
    return closest_city_tile, closest_dist

game_state = None

def get_random_step():
    return np.random.choice(['s','n','w','e'])


def agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    resource_tiles = find_resources(game_state)
        
    # max number of units available
    units_cap = sum([len(x.citytiles) for x in player.cities.values()])
    # current number of units
    units  = len(player.units)
    
    cities = list(player.cities.values())
    if len(cities) > 0:
        city = cities[0]
        created_worker = (units >= units_cap)
        for city_tile in city.citytiles[::-1]:
            if city_tile.can_act():
                if created_worker:
                    # let's do research
                    action = city_tile.research()
                    actions.append(action)
                else:
                    # let's create one more unit in the last created city tile if we can
                    action = city_tile.build_worker()
                    actions.append(action)
                    created_worker = True
    
    
    # we want to build new tiless only if we have a lot of fuel in all cities
    can_build = True
    night_steps_left = ((359 - observation["step"]) // 40 + 1) * 10
    for city in player.cities.values():            
        if city.fuel / (city.get_light_upkeep() + 20) < min(night_steps_left, 30):
            can_build = False
       
    steps_until_night = 30 - observation["step"] % 40
    
    
    # we will keet all tiles where any unit wants to move in this set to avoid collisions
    taken_tiles = set()
    for unit in player.units:
        # it is too strict but we don't allow to go to the the currently occupied tile
        taken_tiles.add((unit.pos.x, unit.pos.y))
        
    for city in opponent.cities.values():
        for city_tile in city.citytiles:
            taken_tiles.add((city_tile.pos.x, city_tile.pos.y))
    
    # we can collide in cities so we will use this tiles as exceptions
    city_tiles = {(tile.pos.x, tile.pos.y) for city in player.cities.values() for tile in city.citytiles}
    
    
    for unit in player.units:
        if unit.can_act():
            closest_resource_tile, closest_resource_dist = find_closest_resources(unit.pos, player, resource_tiles)
            closest_city_tile, closest_city_dist = find_closest_city_tile(unit.pos, player)
            
            # we will keep possible actions in a priority order here
            directions = []
            
            # if we can build and we are near the city let's do it
            if unit.is_worker() and unit.can_build(game_state.map) and ((closest_city_dist == 1 and can_build) or (closest_city_dist is None)):
                # build a new cityTile
                action = unit.build_city()
                actions.append(action)  
                can_build = False
                continue
            
            # base cooldown for different units types
            base_cd = 2 if unit.is_worker() else 3
            
            # how many steps the unit needs to get back to the city before night (without roads)
            steps_to_city = unit.cooldown + base_cd * closest_city_dist
            
            # if we are far from the city in the evening or just full let's go home
            if (steps_to_city + 3 > steps_until_night or unit.get_cargo_space_left() == 0) and closest_city_tile is not None:
                actions.append(annotate.line(unit.pos.x, unit.pos.y, closest_city_tile.pos.x, closest_city_tile.pos.y))
                directions = [unit.pos.direction_to(closest_city_tile.pos)]
            else:
                # if there is no risks and we are not mining resources right now let's move toward resources
                if closest_resource_dist != 0 and closest_resource_tile is not None:
                    actions.append(annotate.line(unit.pos.x, unit.pos.y, closest_resource_tile.pos.x, closest_resource_tile.pos.y))
                    directions = [unit.pos.direction_to(closest_resource_tile.pos)]
                    # optionally we can add random steps
                    for _ in range(2):
                        directions.append(get_random_step())

            moved = False
            for next_step_direction in directions:
                next_step_position = unit.pos.translate(next_step_direction, 1)
                next_step_coordinates = (next_step_position.x, next_step_position.y)
                # make only moves without collision
                if next_step_coordinates not in taken_tiles or next_step_coordinates in city_tiles:
                    action = unit.move(next_step_direction)
                    actions.append(action)
                    taken_tiles.add(next_step_coordinates)
                    moved = True
                    break
            
            if not moved:
                # if we are not moving the tile is occupied
                taken_tiles.add((unit.pos.x,unit.pos.y))
    return actions