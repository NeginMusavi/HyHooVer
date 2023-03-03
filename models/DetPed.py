from __future__ import print_function
import argparse
import collections
import glob
import logging
import math
import os
import random
import sys
import weakref
import time

import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils

import numpy as np
import carla
from carla import ColorConverter as cc

# from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
# from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from yolo_agent import *
from ego_agent import *

import torch
YOLO_base = '/home/carla/YOLOv3/'
# sys.path.insert(0, YOLO_base)
sys.path.append('/home/carla/')
from YOLOv3.models import *
from YOLOv3.utils.utils import *
from YOLOv3.utils.transforms import *
del sys.path[-1]
import torchvision.transforms as transforms

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

latest_other_car_cam_image = None
# rotation = carla.Rotation(pitch=180, roll=180, yaw=0)

# def set_rot(r):
#     global rotation
#     rotation = r

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, loc_ego):
        """Constructor method"""
        self.world = carla_world
        self.map = self.world.get_map()
        self.players = []
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_index = 0
        self._gamma = 2.2
        self.restart(loc_ego)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, loc_ego):
        """Restart the world"""
        global latest_other_car_cam_image#, rotation
        if len(self.players) > 0:
            self.destroy()

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        loc_ped = np.array([390, 137.364777, 0.285304])
        yaw_ped = 0.234757
        transform = carla.Transform(carla.Location(x=loc_ped[0], y=loc_ped[1], z=loc_ped[2]),
                                    carla.Rotation(yaw=yaw_ped))
        # print("Spawning the walker")
        walker_bp = self.world.get_blueprint_library().filter("walker.*")
        walker = self.world.spawn_actor(random.choice(walker_bp), transform)

        yaw_ego = 0.234757
        transform = carla.Transform(carla.Location(x=loc_ego[0], y=loc_ego[1], z=loc_ego[2]), carla.Rotation(yaw=yaw_ego))
        # print("Spawning the ego car. It is using BasicAgent class")
        car_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        car_bp.set_attribute('role_name', 'hero')
        if car_bp.has_attribute('color'):
            # color = random.choice(car_bp.get_attribute('color').recommended_values)
            color = car_bp.get_attribute('color').recommended_values[5]
            # print(car_bp.get_attribute('color').recommended_values)
            car_bp.set_attribute('color', color)
        # transform = spawn_points[208]
        ego_car = self.world.try_spawn_actor(car_bp, transform)

        if not hasattr(self, 'spwan_points'):
            self.spawn_points = self.map.get_spawn_points()
        transform = self.spawn_points[203]
        # print("Spawning the other car. It is on autopilot mode")
        car_bp = self.world.get_blueprint_library().find('vehicle.nissan.patrol')
        car_bp.set_attribute('role_name', 'other')
        if car_bp.has_attribute('color'):
            # color = random.choice(car_bp.get_attribute('color').recommended_values)
            color = car_bp.get_attribute('color').recommended_values[1]
            car_bp.set_attribute('color', color)
        other_car = self.world.try_spawn_actor(car_bp, transform)
        self.players = [ego_car, other_car, walker]

        blp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.other_car_cam = self.players[1].get_world().spawn_actor(
            blp, carla.Transform(
            carla.Location(x=1, z=1.8), carla.Rotation(pitch=270, roll=180, yaw=0)),
            attach_to=self.players[1],
            attachment_type=carla.AttachmentType.SpringArm)
        def image_handler(image):
            global latest_other_car_cam_image
            i = np.array(image.raw_data)
            i2 = i.reshape((image.height, image.width, 4))
            i3 = i2[:, :, :3]
            latest_other_car_cam_image = i3

        self.other_car_cam.listen(image_handler)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.players[0])
        self.lane_invasion_sensor = LaneInvasionSensor(self.players[0])
        self.gnss_sensor = GnssSensor(self.players[0])
        self.camera_manager = CameraManager(self.players[0], self._gamma)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)

    def tick(self):
        """Method for every tick"""
        self.world.tick()

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.players[0],
            self.players[1],
            self.players[2],
            self.other_car_cam]
        for actor in actors:
            if actor is not None:
                actor.destroy()

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, gamma_correction):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                if blp.has_attribute('gamma'):
                    blp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game loop --------------------------------------------------------
# ==============================================================================

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)
np.random.seed(1024)

world = client.load_world('Town06')

pos_x = 302  # [292-302]
loc_ego = np.array([pos_x, 137.284500, 0.285304])
world = World(world, loc_ego)

traffic_manager = client.get_trafficmanager(8000)

settings = world.world.get_settings()
traffic_manager.set_synchronous_mode(True)
settings.fixed_delta_seconds = 0.05
world.world.apply_settings(settings)

loc_ped = np.array([390, 137.364777, 0.285304])
yaw_ped = 0.234757
transform_walker_start = carla.Transform(carla.Location(x=loc_ped[0], y=loc_ped[1], z=loc_ped[2]),
                                         carla.Rotation(yaw=yaw_ped))
world.world.debug.draw_string(transform_walker_start.location, 'walker', draw_shadow=False,
                              color=carla.Color(r=255, g=0, b=0), life_time=1e6,
                              persistent_lines=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up model
img_size = 416
model = Darknet(YOLO_base + 'config/yolov3.cfg', img_size=img_size).to(device)
model.load_darknet_weights(YOLO_base + 'weights/yolov3.weights')
model.eval()  # Set in evaluation mode
classes = load_classes(YOLO_base + 'data/coco.names')  # Extracts class labels from file
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# Bounding-box colors
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
transform = transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)])

def detect(image):
    # Configure input
    image_torch, _ = transform((image, np.zeros((1, 5))))
    image_torch = image_torch.type(Tensor).unsqueeze(0)

    with torch.no_grad():
        detections = model(image_torch)
        detections = non_max_suppression(detections, 0.8, 0.4)

    detected = False
    detections = detections[0]
    image = imutils.resize(image, width=image.shape[1])
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, img_size, image.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections.cpu().detach().numpy().tolist():
            if classes[int(cls_pred)] != 'person':
                continue
            detected = True
            h = image_torch.shape[0]
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(image, classes[int(cls_pred)], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,255], 2)
    # cv2.imshow("", image)
    cv2.waitKey(1)

    return detected

def simulate(init_):

    init = init_[1:]

    step_in_distance, extra_speed = init
    pos_x = 302
    loc_ego = np.array([pos_x, 137.284500, 0.285304])
    world.restart(loc_ego)
    t = 0
    max_t = 100
    walker_coll = 3
    vehicle_coll = 4.7
    walker_speed = 3
    stop = False
    walker_stop = False

    settings = world.world.get_settings()
    traffic_manager.set_synchronous_mode(True)
    settings.fixed_delta_seconds = 0.05
    # settings.no_rendering_mode = True
    world.world.apply_settings(settings)

    speed_limit = world.players[0].get_speed_limit()
    target_speed = speed_limit
    agent_ego = EGOAgent(world.players[0], target_speed)
    destination = np.array([490.693512, 137.585999, 0.300000])
    agent_ego.set_destination(destination)

    speed_limit = world.players[1].get_speed_limit()
    target_speed = speed_limit + extra_speed
    agent_other = YOLOAgent(world.players[1], target_speed)
    destination = np.array([490.693512, 137.585999, 0.300000])
    agent_other.set_destination(destination)


    walker_control = carla.WalkerControl()
    walker_heading = 90
    walker_rotation = carla.Rotation(0, walker_heading, 0)
    walker_control.direction = walker_rotation.get_forward_vector()
    walker_control.speed = 0

    while True:
        spectator = world.world.get_spectator()
        # agent.update_information()
        world.tick()

        spectator.set_transform(carla.Transform(world.players[1].get_location() + carla.Location(z=30),
                                                carla.Rotation(pitch=-90)))

        distance_1 = world.players[1].get_location().distance(world.players[2].get_location())
        distance_2 = world.players[1].get_location().distance(world.players[0].get_location())
        distance_x = world.players[2].get_location().x - world.players[1].get_location().x
        # print('distance of the other car and the walker:', distance_1)
        # print('distance of the other car and the ego car:', distance_2)
        # print('vel:', world.players[1].get_velocity())
        if distance_x <= step_in_distance:
            walker_control.speed = walker_speed

        if distance_1 <= walker_coll:
            walker_stop = True
        if walker_stop:
            walker_control.speed = 0

        walker_list = []
        walker_detected = detect(latest_other_car_cam_image)
        if walker_detected:
            walker_list.append(world.players[2])
        control = agent_ego.run_step()
        world.players[0].apply_control(control)
        control, stop = agent_other.run_step(walker_detected, walker_list, stop, distance_1)
        world.players[1].apply_control(control)
        world.players[2].apply_control(walker_control)

        t = t + 1
        # print(t)

        if t >= max_t or distance_2 <= vehicle_coll:
            if t >= max_t:
                print("time horizon over!")
            else:
                # print("t:", t)
                print("t:", t, 'unsafe!')
                return True
            break
    return False

import numpy as np
import sys
sys.path.append('..')
from NiMC import NiMC

__all__ = ['CARLA_2Car1PedYOLODet']

class CARLA_2Car1PedYOLODet(NiMC):
    def __init__(self, k=0):
        super(CARLA_2Car1PedYOLODet, self).__init__()

        Theta_ = [[12, 30], [-10, 10]]
        categories = [0]

        Theta = []
        Theta.append(categories)
        for i in range(len(Theta_)):
            Theta.append(Theta_[i])

        self.set_Theta(Theta)
        self.set_k(k)

    def is_unsafe(self, state):
        return simulate(state)

    def transition(self, state):
        assert len(state) == self.Theta.shape[0]
        return state
