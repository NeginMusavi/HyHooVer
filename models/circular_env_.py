from typing import Tuple
from gym.envs.registration import register
import numpy as np


from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane, AbstractLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.objects import Obstacle


class CircularEnv(AbstractEnv):

    COLLISION_REWARD: float = -1
    HIGH_SPEED_REWARD: float = 0.2
    RIGHT_LANE_REWARD: float = 0
    LANE_CHANGE_REWARD: float = -0.05

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "absolute": True,
                "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-15, 15], "vy": [-15, 15]},
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "incoming_vehicle_destination": None,
            "screen_width": 500,
            "screen_height": 500,
            "centering_position": [0.6, 0.65],
            "duration": 11
        })
        return config

    def _reward(self, action: int) -> float:
        lane_change = action == 0 or action == 2
        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.HIGH_SPEED_REWARD * MDPVehicle.get_speed_index(self.vehicle) / max(MDPVehicle.SPEED_COUNT - 1, 1) \
                 + self.LANE_CHANGE_REWARD * lane_change
        return utils.lmap(reward, [self.COLLISION_REWARD + self.LANE_CHANGE_REWARD, self.HIGH_SPEED_REWARD], [0, 1])

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.steps >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:

        # ----------------------------Explanation--------------------------------
        # -----------------------------------------------------------------------
        # This network consists of a circular intersection with 4 (two way) access lanes,

        # The circular intersection has 4 access lanes.
        # Access lanes are tow way lanes that merges to the intersection with sine lanes.
        # Notations: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        # For instance, "se" and "sx" stand for south entry and south exist (used for sine lanes), respectively.
        # Other notations: "r" and "s" ae added to (for instance) "se", which result in "ser" and "sex".
        # These are used to define start to start and end position of access lanes.
        # "r" is for distant location from the intersection and,
        # "s" is for closet location to the intersection.

        # ----------------------------------------------------------------------

        # ----------------------------Parameters--------------------------------
        # ----------------------------------------------------------------------
        access = 200  # [m] length of the access lanes
        # --------for merging lane------------------
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2 * dev  # [m]
        delta_en = dev - delta_st
        w = 2 * np.pi / dev
        ends = [150, 80, 80, access]  # Before, converging, merge, after
        y = StraightLane.DEFAULT_WIDTH
        # -------------------------------------------
        # --------for circular intersection-------------
        radius = 20  # [m]
        center1 = [sum(ends) + 2 * radius, y - y/2]  # [m]
        alpha = 24  # [deg]
        radii = [radius, radius + y]
        # ----------------------------------------------------------------------

        # ----------------------------Create the Network------------------------
        # ----------------------------------------------------------------------
        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        # ---- Circular Intersection ----
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("se", "ex",
                         CircularLane(center1, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ex", "ee",
                         CircularLane(center1, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ee", "nx",
                         CircularLane(center1, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("nx", "ne",
                         CircularLane(center1, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ne", "wx",
                         CircularLane(center1, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("wx", "we",
                         CircularLane(center1, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("we", "sx",
                         CircularLane(center1, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("sx", "se",
                         CircularLane(center1, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha),
                                      clockwise=False, line_types=line[lane]))

        # ---- West Access Lanes ---- (entry)
        net.add_lane("l1", "r1", StraightLane([0, 2 + center1[1]], [sum(ends[:2]), 2 + center1[1]], line_types=(s, c)))
        net.add_lane("r1", "wer",
                     StraightLane([sum(ends[:2]), 2 + center1[1]], [sum(ends[:3]), 2 + center1[1]], line_types=(s, c)))
        net.add_lane("wer", "wes",
                     StraightLane([sum(ends[:3]), 2 + center1[1]], [sum(ends), 2 + center1[1]], line_types=(s, c)))
        net.add_lane("wes", "we", SineLane([-dev / 2 + center1[0], 2 + a + center1[1]],
                                           [-dev / 2 + delta_st + center1[0], 2 + a + center1[1]], a, w, -np.pi / 2,
                                           line_types=(c, c)))


        # ---- West Access Lanes ---- (exit)
        net.add_lane("r2", "l2",
                     StraightLane([sum(ends[:2]), -2 + center1[1]], [0, -2 + center1[1]], line_types=(n, c)))
        net.add_lane("wxr", "r2", StraightLane([sum(ends[:3]), -2 + center1[1]], [sum(ends[:2]), -2 + center1[1]],
                                               line_types=(n, c)))
        net.add_lane("wxs", "wxr",
                     StraightLane([sum(ends), -2 + center1[1]], [sum(ends[:3]), -2 + center1[1]], line_types=(n, c)))
        net.add_lane("wx", "wxs", SineLane([dev / 2 - delta_en + center1[0], -2 - a + center1[1]],
                                           [-dev / 2 + center1[0], -2 - a + center1[1]], a, w,
                                           -np.pi / 2 + w * delta_en, line_types=(c, c)))

        # ---- South Access Lane ----
        net.add_lane("ser", "ses",
                     StraightLane([2 + center1[0], access + center1[1]], [2 + center1[0], dev / 2 + center1[1]],
                                  line_types=(s, c)))
        net.add_lane("ses", "se", SineLane([2 + a + center1[0], dev / 2 + center1[1]],
                                           [2 + a + center1[0], dev / 2 - delta_st + center1[1]], a, w, -np.pi / 2,
                                           line_types=(c, c)))
        net.add_lane("sx", "sxs", SineLane([-2 - a + center1[0], -dev / 2 + delta_en + center1[1]],
                                           [-2 - a + center1[0], dev / 2 + center1[1]], a, w, -np.pi / 2 + w * delta_en,
                                           line_types=(c, c)))
        net.add_lane("sxs", "sxr",
                     StraightLane([-2 + center1[0], dev / 2 + center1[1]], [-2 + center1[0], access + center1[1]],
                                  line_types=(n, c)))

        # ---- North Access Lane ----
        net.add_lane("ner", "nes",
                     StraightLane([-2 + center1[0], -access + center1[1]], [-2 + center1[0], -dev / 2 + center1[1]],
                                  line_types=(s, c)))
        net.add_lane("nes", "ne", SineLane([-2 - a + center1[0], -dev / 2 + center1[1]],
                                           [-2 + center1[0] - a, -dev / 2 + delta_st + center1[1]], a, w, -np.pi / 2,
                                           line_types=(c, c)))
        net.add_lane("nx", "nxs", SineLane([2 + a + center1[0], dev / 2 - delta_en + center1[1]],
                                           [2 + center1[0] + a, -dev / 2 + center1[1]], a, w, -np.pi / 2 + w * delta_en,
                                           line_types=(c, c)))
        net.add_lane("nxs", "nxr",
                     StraightLane([2 + center1[0], -dev / 2 + center1[1]], [2 + center1[0], -access + center1[1]],
                                  line_types=(n, c)))

        # ---- East Access Lane ----
        net.add_lane("eer", "ees",
                     StraightLane([access + center1[0], -2 + center1[1]], [dev / 2 + center1[0], -2 + center1[1]],
                                  line_types=(s, c)))
        net.add_lane("ees", "ee", SineLane([dev / 2 + center1[0], -2 - a + center1[1]],
                                           [dev / 2 + center1[0] - delta_st, -2 - a + center1[1]], a, w, -np.pi / 2,
                                           line_types=(c, c)))
        net.add_lane("ex", "exs", SineLane([-dev / 2 + center1[0] + delta_en, 2 + a + center1[1]],
                                           [dev / 2 + center1[0], 2 + a + center1[1]], a, w, -np.pi / 2 + w * delta_en,
                                           line_types=(c, c)))
        net.add_lane("exs", "exr",
                     StraightLane([dev / 2 + center1[0], 2 + center1[1]], [access + center1[0], 2 + center1[1]],
                                  line_types=(n, c)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate several vehicles on the lanes.

        :return: the ego-vehicle
        """

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        # # ----------for observation----------
        # ego_vehicle = self.action_type.vehicle_class(self.road,
        #                                              self.road.network.get_lane(("l1", "r1", 0)).position(100, 0),
        #                                              speed=10)
        # try:
        #     ego_vehicle.plan_route_to("s1")
        # except AttributeError:
        #     pass
        #
        # self.road.vehicles.append(ego_vehicle)
        # self.vehicle = ego_vehicle

        # --------circular intersection part---------
        ego_lane = self.road.network.get_lane(("ser", "ses", 0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(155, 0),
                                                     speed=10,
                                                     heading=ego_lane.heading_at(270))
        # try:
        #     ego_vehicle.plan_route_to("nxs") # this can be "exs", "nxs", "wxs"
        # except AttributeError:
        #     pass
        ego_vehicle.target_speed = 10
        self.road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        #--------------------------------------------------------------------------
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("ser", "ses", 0),
                                                   longitudinal=120,
                                                   speed=10)
        vehicle.plan_route_to("wxr")
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        #--------------------------------------------------------------------------
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("we", "sx", 0),
                                                   longitudinal=0,
                                                   speed=10)
        # vehicle.plan_route_to("exr") # this an be "exr", "nxr"
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        #--------------------------------------------------------------------------
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("we", "sx", 0),
                                                   longitudinal=25,
                                                   speed=12)
        vehicle.plan_route_to("exr")
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)


register(
    id='circular-v0',
    entry_point='highway_env.envs:CircularEnv',
)