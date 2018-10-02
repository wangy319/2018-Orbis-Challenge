from PythonClientAPI.game.PointUtils import *
from PythonClientAPI.game.Entities import FriendlyUnit, EnemyUnit, Tile
from PythonClientAPI.game.Enums import Team, Direction
from PythonClientAPI.game.World import World
from PythonClientAPI.game.TileUtils import TileUtils

import numpy as np

class PlayerAI:

    def __init__(self):
        ''' Initialize! '''
        self.turn_count = 0             # game turn count
        self.target = None              # target to send unit to!
        self.outbound = True            # is the unit leaving, or returning?
        self.inside = True              # is the unit inside the territory?

    def get_direction_vector(self,start,end):
        return (end[0]-start[0],end[1]-start[1])

    def get_normalized_direction_vector(self,start,end):
        return ((end[0]-start[0])/max(abs(end[0]-start[0]),abs(end[1]-start[1])),(end[1]-start[1])/max(abs(end[0]-start[0]),abs(end[1]-start[1])))


    def do_move(self, world, friendly_unit, enemy_units):
        '''
        This method is called every turn by the game engine.
        Make sure you call friendly_unit.move(target) somewhere here!

        Below, you'll find a very rudimentary strategy to get you started.
        Feel free to use, or delete any part of the provided code - Good luck!

        :param world: world object (more information on the documentation)
            - world: contains information about the game map.
            - world.path: contains various pathfinding helper methods.
            - world.util: contains various tile-finding helper methods.
            - world.fill: contains various flood-filling helper methods.

        :param friendly_unit: FriendlyUnit object
        :param enemy_units: list of EnemyUnit objects

        '''
        # mode list:
        #   explore:
        #       inside:
        #           always try to move towards the second farthest grid corner
        #           also try to move towards the closest territory edge (to get out)
        #       outside:
        #           always try to keep a certain distance with the nearest territory edge
        #           move away from the center of snake
        #           move towards the center of non-territory space
        #
        #   defense
        #		Calculate body to enemy's head

        # direction heuristic dictionary
        heuristic_dict = {Direction.EAST: 0, Direction.NORTH: 0, Direction.WEST: 0, Direction.SOUTH: 0}
        direction_heuristic = [0,0] # x and y axis
        # init heuristic
        h = np.zeros((7, 2))
        weight = np.ones((7,))
        # weight = np.array([0.05,4,5,1,1,1])

        # world grid corners
        nw = (0,0)
        sw = (0,world.get_height()-1)
        ne = (world.get_width()-1,0)
        se = (world.get_width()-1,world.get_height()-1)
        corners = [nw, ne, sw, se]
        # current status
        curr_pos = friendly_unit.position # tuple
        curr_terr = friendly_unit.territory # set of tuple
        curr_snake = friendly_unit.snake # set of tuple
        curr_length = len(curr_snake)
        grid = set([(x,y) for x in range(world.get_width()-1) for y in range(world.get_height()-1)])


        # determine current mode
        if curr_pos in curr_terr:
            self.inside = True
        else:
            self.inside = False

        # if unit is dead, stop making moves.
        if friendly_unit.status == 'DISABLED':
            print("Turn {0}: Disabled - skipping move.".format(str(self.turn_count)))
            self.target = None
            self.outbound = True
            return

        # Defense
        # Avoid body being eaten by enemy's heads
        # Find Other enemy's heads position -> find the closet one
        if not len(friendly_unit.body) == 0:
            closest_enemy_dist = [world.path.get_shortest_path_distance(enemy.position, world.util.get_closest_friendly_body_from(enemy.position, None).position) for enemy in enemy_units if world.util.get_closest_friendly_body_from(enemy.position, None) is not None]
            closest_body_posi = (world.util.get_closest_friendly_territory_from(curr_pos, curr_snake)).position
            closest_posi_dist = world.path.get_shortest_path_distance(curr_pos, closest_body_posi)

            if min(closest_enemy_dist) <= closest_posi_dist+2:
                friendly_unit.move(world.path.get_shortest_path(curr_pos, closest_body_posi, curr_snake)[0])
                return


        if self.inside:
            # heuristic 1
            # calculate distance to 4 grid corners and get the shortest path to it
            # move towards the the second farthest grid corner
            curr_terr_center = tuple((int(np.mean(np.array(list(curr_terr)), axis=0)[0]),int(np.mean(np.array(list(curr_terr)), axis=0)[1])))
            distance_to_corners = []
            for corner in corners:
                distance_to_corners.append(tuple((corner, world.path.get_taxi_cab_distance(curr_terr_center,corner))))
            distance_to_corners = sorted(distance_to_corners, key=lambda dist: dist[1])
            if world.path.get_taxi_cab_distance(curr_pos, distance_to_corners[2][0]) < world.path.get_taxi_cab_distance(curr_pos, distance_to_corners[1][0]):
                h[0] = np.array(self.get_direction_vector(curr_pos, distance_to_corners[2][0]))
            else:
                h[0] = np.array(self.get_direction_vector(curr_pos, distance_to_corners[1][0]))
            weight[0] = 0.05

            # heuristic 2
            # also try to move towards the closest territory edge (to get out)
            closest_edge_posi = (world.util.get_closest_capturable_territory_from(friendly_unit.position, None)).position
            h[1] = self.get_direction_vector(curr_pos, closest_edge_posi)
            weight[1] = 4

        else:
            # heuristic 3
            # always try to keep a certain distance with the nearest territory edge
            closest_posi = (world.util.get_closest_friendly_territory_from(curr_pos, None)).position
            if world.path.get_shortest_path_distance(curr_pos,closest_posi) <= min(6, 20 - curr_length):
                h[2] = self.get_normalized_direction_vector(closest_posi,curr_pos)
                weight[2] = 5.0/curr_length
            else:
                h[2] = self.get_normalized_direction_vector(curr_pos,closest_posi)
                weight[2] = 2 * curr_length

            # heuristic 4
            # move away from the center of snake
            curr_snake_center = tuple((int(np.mean(np.array(list(curr_snake)), axis=0)[0]),int(np.mean(np.array(list(curr_snake)), axis=0)[1])))
            h[3] = self.get_direction_vector(curr_snake_center,curr_pos)
            weight[3] = 2

            # heuristic 5
            # move towards the center of non-territory space
            # non_terr = grid - curr_terr
            # curr_non_terr_center = tuple((int(np.mean(np.array(list(non_terr)), axis=0)[0]),
            #                            int(np.mean(np.array(list(non_terr)), axis=0)[1])))
            # h[4] = self.get_direction_vector(curr_pos,curr_non_terr_center)

            # heuristic 5
            # move away form the center of the territory space
            # curr_terr_center = tuple((int(np.mean(np.array(list(curr_terr)), axis=0)[0]),int(np.mean(np.array(list(curr_terr)), axis=0)[1])))
            # h[4] = self.get_direction_vector(curr_terr_center, curr_pos)

        # heuristic 5
        # move towards the closest position of an enemy body
        closest_enemy_body = world.util.get_closest_enemy_body_from(curr_pos, friendly_unit.snake)
        if not closest_enemy_body == None:
            h[4] = self.get_normalized_direction_vector(curr_pos, closest_enemy_body.position)
            weight[4] = 10.0 / np.sqrt(world.path.get_shortest_path_distance(curr_pos, closest_enemy_body.position))

        # heuristic 6
        # move towards the opposite grid corner
        curr_terr_center = tuple((int(np.mean(np.array(list(curr_terr)), axis=0)[0]),
                                  int(np.mean(np.array(list(curr_terr)), axis=0)[1])))
        distance_to_corners = []
        for corner in corners:
            distance_to_corners.append(tuple((corner, world.path.get_taxi_cab_distance(curr_terr_center, corner))))
        distance_to_corners = sorted(distance_to_corners, key=lambda dist: dist[1])
        init_target = distance_to_corners[-1][0]
        h[5] = self.get_direction_vector(curr_pos, init_target)
        weight[5] = 0.2

        # heuristic 7
        # Move towards any of the closest enemy's territory to eat its units
        exclusion_list = [enemy.position for enemy in enemy_units] + list(curr_snake)
        closest_enemy_tile = world.util.get_closest_enemy_territory_from(curr_pos, exclusion_list)
        if not closest_enemy_tile == None:
            h[6] = self.get_direction_vector(curr_pos,closest_enemy_tile.position)
            weight[6] = 10/np.square(world.path.get_shortest_path_distance(curr_pos, closest_enemy_tile.position))

        # Calibration of weights
        if self.inside:
            # weight[4] = 3
            # weight[5] = 1
            pass
        else:
            # weight[4] = 3
            # weight[5] = 0.1
            pass

        # Calculate final move target
        print(h)
        print(weight)
        final_direction = np.dot(weight,h)
        print(final_direction)
        prefer_dict = {}
        for direction in Direction.ORDERED_DIRECTIONS:
            prefer_dict[direction] = np.dot(np.array(direction.value),final_direction)
        print(prefer_dict)

        new_pos = None
        for key, value in sorted(prefer_dict.items(), key=lambda kv: kv[1], reverse=True):
            print("the key is: {0} and the value is: {1}".format(key,value))
            new_pos = key.move_point(curr_pos)
            if not world.is_within_bounds(new_pos):
                continue
            if new_pos in curr_snake:
                continue
            if world.is_wall(new_pos ):
                continue
            break

        # Move
        friendly_unit.move(new_pos)

        # increment turn count
        self.turn_count += 1
        print("Turn {0}: currently at {1}, move to {2}.".format(
            str(self.turn_count),
            str(friendly_unit.position),
            str(new_pos)
        ))
