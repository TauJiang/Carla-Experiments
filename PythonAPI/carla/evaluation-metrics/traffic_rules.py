"""
Author: Yingtao Jiang
Reference: Carla
Evaluate if the agent follow general traffic rules
"""

import abc
import logging
import argparse
import math
import time
import random
import numpy as np

import carla

# Traffic Signs
# Determine Nearest Traffic Signs with the correct Orientation State
# Determine violation of the traffic signs

def get_vec_dist(x_dst, y_dst, x_src, y_src):
    vec = np.array([x_dst, y_dst] - np.array([x_src, y_src]))
    dist = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return vec / dist, dist

def is_vehicle_in_intersection(v_location, traffic_light, junction):
    tl_location = junction.bounding_box.location
    tl_extent = junction.bounding_box.extent
    tl_rotation = junction.bounding_box.rotation

    if tl_location.y - tl_extent.y <= v_location.y and \
        tl_location.y + tl_extent.y >= v_location.y and \
        tl_location.x - tl_extent.x <= v_location.x and \
        tl_location.x + tl_extent.x >= v_location.x:
        return True
    return False

def _is_traffic_light_active(self, agent, orientation):
    x_agent = agent.traffic_light.transform.location.x
    y_agent = agent.traffic_light.transform.location.y

    def search_closest_lane_point(x_agent, y_agent, depth):
        step_size = 4
        if depth > 1:
            return None
        try:
            degrees = self._map.get_lane_orientation_degrees([x_agent, y_agent, 38])
            #print (degrees)
        except:
            return None

        if not self._map.is_point_on_lane([x_agent, y_agent, 38]):
            #print (" Not on lane ")
            result = search_closest_lane_point(x_agent + step_size, y_agent, depth+1)
            if result is not None:
                return result
            result = search_closest_lane_point(x_agent, y_agent + step_size, depth+1)
            if result is not None:
                return result
            result = search_closest_lane_point(x_agent + step_size, y_agent + step_size, depth+1)
            if result is not None:
                return result
            result = search_closest_lane_point(x_agent + step_size, y_agent - step_size, depth+1)
            if result is not None:
                return result
            result = search_closest_lane_point(x_agent - step_size, y_agent + step_size, depth+1)
            if result is not None:
                return result
            result = search_closest_lane_point(x_agent - step_size, y_agent, depth+1)
            if result is not None:
                return result
            result = search_closest_lane_point(x_agent, y_agent - step_size, depth+1)
            if result is not None:
                return result
            result = search_closest_lane_point(x_agent - step_size, y_agent - step_size, depth+1)
            if result is not None:
                return result
        else:
            #print(" ON Lane ")
            if degrees < 6:
                return [x_agent, y_agent]
            else:
                return None

    closest_lane_point = search_closest_lane_point(x_agent, y_agent, 0)
    car_direction = math.atan2(orientation.y, orientation.x) + 3.1415
    if car_direction > 6.0:
        car_direction -= 6.0

    return math.fabs(car_direction -
        self._map.get_lane_orientation_degrees([closest_lane_point[0], closest_lane_point[1], 38])
                        ) < 1

def agent_location():
    pass

# Traffic Light
def _test_for_traffic_lights(world, vehicle):
    """

    This function tests if the car passed into a traffic light, returning 'red'
    if it crossed a red light , 'green' if it crossed a green light or none otherwise

    Args:
        

    Returns:

    """

    def is_on_burning_point(vehicle, traffic_light, junction):

        # We get the current lane orientation
        # ori_x, ori_y = _map.get_lane_orientation([location.x, location.y, 38])
        ori = vehicle.get_transform().rotation.yaw
        ori_x = -math.cos(ori)
        ori_y = -math.sin(ori)

        # We test to walk in direction of the lane
        future_location_x = vehicle.get_location().x
        future_location_y = vehicle.get_location().y

        for i in range(3):
            future_location_x += ori_x
            future_location_y += ori_y
        # Take a point on a intersection in the future
        location_on_intersection_x = future_location_x + 2 * ori_x
        location_on_intersection_y = future_location_y + 2 * ori_y

        class VLocation():
            def __init__(self, x, y) -> None:
                self.x = x
                self.y = y
        future_location = VLocation(future_location_x, future_location_y)

        if is_vehicle_in_intersection(future_location, traffic_light, junction) and \
            not is_vehicle_in_intersection(vehicle.get_location(), traffic_light, junction):
            return True

        return False

    # Check nearest traffic light with the correct orientation state.
    player_x = vehicle.get_location().x
    player_y = vehicle.get_location().y
    print(player_x, player_y)
    world.debug.draw_point(vehicle.get_transform().location)
    # world.debug.draw_point(vehicle.get_location(), color=carla.Color(100,100,0))
    # print(vehicle.attributes)
    # print(vehicle.semantic_tags)
    # print(world.get_actors())
    for agent in world.get_actors():
        if agent.type_id == 'traffic.traffic_light':
            color = carla.Color(random.randint(0, 250), random.randint(0, 250), random.randint(0, 250))
            world.debug.draw_string(agent.get_location(), 'o', color=color, life_time = 1)
            world.debug.draw_point(agent.trigger_volume.location, color=color)
            
            agent_location = agent.trigger_volume.location
            temp_location = carla.Location(x= agent.get_location().x + agent_location.x, 
                                            y= agent.get_location().y + agent_location.y, 
                                            z= agent.get_location().z + agent_location.z)
            temp_location = carla.Location(x= agent.get_location().x, 
                                            y= agent.get_location().y, 
                                            z= agent.get_location().z)
            volume = carla.BoundingBox(temp_location, agent.trigger_volume.extent)
            world.debug.draw_point(agent.get_location(), color=carla.Color(100,10,200)) 
            # world.debug.draw_box(volume, agent.trigger_volume.rotation, color = color)

            waypoints = agent.get_affected_lane_waypoints()
            junction = 0
            for waypoint in waypoints:
                world.debug.draw_point(waypoint.transform.location, color=color)
                if waypoint.is_junction:
                    print('in junction')
                    # print(waypoint.get_junction())
                    junction = waypoint.get_junction()
                    # print(junction.id, junction.bounding_box)
                    world.debug.draw_box(junction.bounding_box, junction.bounding_box.rotation, color=color)
                    junction_waypoints = junction.get_waypoints(carla.LaneType.Any)
                    # print(junction_waypoints)
                    for junction_waypoint2 in junction_waypoints:
                        for junction_waypoint in junction_waypoint2:
                            world.debug.draw_point(junction_waypoint.transform.location, color=color)
                    break
            # print(junction.id, junction.bounding_box)
            if not is_vehicle_in_intersection(vehicle.get_location(), agent, junction):
                x_agent = agent.get_location().x
                y_agent = agent.get_location().y
                tl_vector, tl_dist = get_vec_dist(x_agent, y_agent, player_x, player_y)
                # if _is_traffic_light_active(agent, vehicle.get_transform().location):
                if True:
                    print("in intersection", tl_dist, is_on_burning_point(vehicle, agent, junction))
                    if is_on_burning_point(vehicle, agent, junction) and tl_dist <6.0:
                        if agent.get_state() == "Red":
                            return 'red'
                        elif agent.get_state() == "Green":
                            return 'green'
                        elif agent.get_state() == "Yellow":
                            return 'yellow'
                        else:
                            return 'unknown'


    # The vehicle is on an intersection so we verify if it burned a traffic light
    # for agent in measurement.non_player_agents:
    #     if agent.HasField('traffic_light'):
    #         if not self._map.is_point_on_intersection([player_x, player_y, 38]):
    #             x_agent = agent.traffic_light.transform.location.x
    #             y_agent = agent.traffic_light.transform.location.y
    #             tl_vector, tl_dist = get_vec_dist(x_agent, y_agent, player_x, player_y)
    #             if self._is_traffic_light_active(agent,
    #                                                 measurement.player_measurements.
    #                                                         transform.orientation):
    #                 if is_on_burning_point(self._map,
    #                                         measurement.player_measurements.transform.location)\
    #                         and tl_dist < 6.0:
    #                     if agent.traffic_light.state != 0:  # Not green
    #                         return 'red'
    #                     else:
    #                         return 'green'

    return None

# Trafffic Speed Limits
# Check speed limit
def testing():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=30,
        type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=10,
        type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        default=False,
        help='Activate no rendering mode')

    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []

    # Carla setup: enable recording
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    print(world)

    blueprint_library = world.get_blueprint_library()
    # print(blueprint_library)
    vehicle_bp = blueprint_library.filter(args.filter)[0]
    vehicle_transform = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
    vehicle.set_autopilot(True)
    # vehicle.show_debug_telemetry()
    print(vehicle)

    step = 0
    while True:
        # measurements, sensor_data = client.read_data()
        # test if car crossed the traffic light
        if step % 10000000 == 0:
            traffic_light_state = _test_for_traffic_lights(world, vehicle)
            print(traffic_light_state)


        step += 1

testing()
