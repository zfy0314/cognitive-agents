import json
import logging
import os
import pdb  # noqa
import random
from collections import deque
from pprint import pformat
from time import sleep
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pygame
from ai2thor import controller
from ai2thor.server import Event

import env.utils as utils
from env.utils import Pos2D, Pos4D

gripper_name = "RobotGripper"
gripper_type = "Gripper"


class AffordanceError(Exception):
    pass


class Env:
    controller: controller.Controller
    event: Event
    interval: float = 0.001
    long_interval: float = 0.3

    def __init__(
        self,
        floorplan: str = "FloorPlan3",
        width: int = 400,
        height: int = 400,
        object_pose_file: Optional[str] = None,
        **kwargs,
    ):

        self.width = width
        self.height = height
        self.floorplan = floorplan
        self.grid_size = 0.1
        self.rotate_step = 30

        self.controller = controller.Controller()

        pygame.init()
        gap = width // 20
        self.screen_width = 3 * width + 2 * gap
        self.screen_height = height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.rgb_top_left = (width + gap, 0)
        self.obj_top_left = (0, 0)
        self.bird_top_left = (2 * (width + gap), 0)

        self.object_pose_file = object_pose_file

        self.reset()

        logging.info("environment started")
        logging.debug("all objects: \n {}".format(pformat(self.objects, indent=2)))

    @property
    def objects_(self) -> List[Dict[str, Any]]:
        return [
            obj_info
            for obj_info in self.event.metadata["objects"]
            if not self._is_ignored(obj_info)
        ]

    @property
    def agent_(self) -> Dict[str, Any]:
        return self.event.metadata["agent"]

    @property
    def objects(self) -> List[str]:
        """Return all object ids in the environment"""
        return sorted([obj_info["name"] for obj_info in self.objects_])

    @property
    def objects_visible(self) -> List[str]:
        """Return all visible object ids in the environment"""

        visibles = []
        for obj_info in self.objects_:
            if self._is_visible(obj_info):
                visibles.append(obj_info["name"])
        return sorted(visibles)

    @property
    def objects_visible_(self) -> List[Dict[str, Any]]:

        visibles = []
        for obj_info in self.objects_:
            if self._is_visible(obj_info):
                visibles.append(obj_info)
        return visibles

    @property
    def object_types_visible(self) -> List[str]:

        return sorted(
            obj_info["objectType"]
            for obj_info in self.objects_
            if self._is_visible(obj_info)
        )

    @property
    def reachable_positions(self) -> List[Dict[str, float]]:
        return self.controller.step(action="GetReachablePositions").metadata[
            "actionReturn"
        ]

    @property
    def object_in_hand(self) -> Optional[Dict[str, Any]]:

        try:
            object_in_hand = self.event.metadata["inventoryObjects"][0]
        except IndexError:
            return None
        else:
            return object_in_hand

    @property
    def location_matrix(self) -> Dict[str, Dict[str, float]]:

        return {
            name1: {name2: loc1 - loc2 for name2, loc2 in self.locations.items()}
            for name1, loc1 in self.locations.items()
        }

    def reset(self):

        location_file = os.path.join("locations/{}.json".format(self.floorplan))
        if os.path.isfile(location_file):
            with open(location_file, "r") as fin:
                locations = json.load(fin)
        else:
            logging.error(
                "No location file found for {} (looked for {})".format(
                    self.floorplan, location_file
                )
            )
            locations = {}
        self.controller.step(action="DropHandObject")
        self.controller.reset(
            scene=self.floorplan,
            width=self.width,
            height=self.height,
            gridSize=self.grid_size,
            rotateStepDegrees=self.rotate_step,
            snapToGrid=False,
            visibilityDistance=1.6,
            fieldOfView=75,
            renderInstanceSegmentation=True,
        )
        if len(locations) > 0:
            default_location = self.current_location = list(locations.keys())[0]
            agent_config = locations[self.current_location]
            agent_config["rotation"]["y"] = utils.angle_round(
                agent_config["rotation"]["y"]
            )
            self.controller.step(
                action="Teleport",
                position=agent_config["position"],
                rotation=agent_config["rotation"],
                horizon=int(agent_config["cameraHorizon"]),
                standing=agent_config["isStanding"],
            )
        self.locations = {
            key: Pos4D(
                info["position"]["x"],
                int(info["cameraHorizon"]),
                info["position"]["z"],
                int(info["rotation"]["y"]),
            )
            for key, info in locations.items()
        }
        self.event = self.controller.step(
            action="AddThirdPartyCamera",
            position=dict(x=0, y=2.8, z=0),
            rotation=dict(x=90, y=0, z=0),
            fieldOfView=100,
        )
        self.navigation_cache = {}

        if self.object_pose_file is not None:
            self.api_step(
                "SetObjectPoses",
                objectPoses=json.load(open(self.object_pose_file)),
                forceAction=True,
            )
            logging.info("setting object poses")
            self._update_screen(self.event)

        self.stacking_info = {}
        for location in sorted(self.locations.keys()):
            if location.startswith("CounterTop") or location.startswith("SinkBasin"):
                self.move_to(location, forbid_same_location=False)
                for obj_info in self.objects_:
                    if self._is_strictly_visible(obj_info):
                        name = obj_info["name"]
                        location = self._get_location(obj_info)
                        if (
                            name not in self.stacking_info.keys()
                            and location is not None
                            and location.startswith("CounterTop")
                        ):
                            logging.debug(
                                "initializing stacking_info[{}] = {}".format(
                                    name, location
                                )
                            )
                            self.stacking_info[name] = location
        self.move_to(default_location)

        self._update_screen(self.event)

    # === low level functions === #

    def api_step(
        self, *args, wait: bool = True, raise_error: bool = True, **kwargs
    ) -> Event:
        if wait:
            sleep(self.interval)

        self.event = self.controller.step(*args, **kwargs)
        if not self.event.metadata["lastActionSuccess"]:
            logging.warning(
                "last action unsuccessful: {}".format(
                    pformat(self.event.metadata["lastAction"], indent=2)
                )
            )
            logging.warning(
                "error message: {}".format(self.event.metadata["errorMessage"])
            )
            if raise_error:
                raise AffordanceError(self.event.metadata["errorMessage"])
        event = self.controller.step(
            action="UpdateThirdPartyCamera",
            position=dict(
                x=self.agent_["position"]["x"],
                y=np.array(self.event.metadata["sceneBounds"]["cornerPoints"])[
                    :, 1
                ].max(),
                z=self.agent_["position"]["z"],
            ),
            rotation=dict(x=90, y=0, z=0),
            fieldOfView=100,
        )
        self._update_screen(event)
        return self.event

    # === helpers === #

    def _teleport_pos4d(self, pos: Pos4D):

        self.api_step(
            action="Teleport",
            position=dict(x=pos.x, y=self.agent_["position"]["y"], z=pos.z),
            rotation=dict(x=0, y=pos.theta, z=0),
            horizon=pos.y,
            standing=True,
            forceAction=True,
        )

    def _init_locations(self):

        locations: Dict[str, Pos4D] = {}

        reachables = self.reachable_positions

        # generate locations for interacting with openable objects
        for obj_info in self.objects_:
            if obj_info["openable"]:
                raise NotImplementedError

        reachable_pos2d = {Pos2D(pos["x"], pos["z"]) for pos in reachables}
        edge_pos: Set[Pos4D] = set()
        edges: List[List[Pos4D]] = []

        # get all the edge positions
        for pos2d in reachable_pos2d:
            for theta, (dx, dz) in utils.orientation_bindings.items():
                dx *= self.grid_size
                dz *= self.grid_size
                neighbor = pos2d + (dx, dz)
                if neighbor not in reachable_pos2d:
                    edge_pos.add(Pos4D(x=neighbor.x, y=15, z=neighbor.z, theta=theta))

        # remove the edge positions that overlaps with existing locations
        to_remove = set()
        for pos4d in edge_pos:
            for existed in locations.values():
                if existed.roughly_equal(pos4d):
                    to_remove.add(pos4d)
                    break
        edge_pos = edge_pos - to_remove

        # generate list of edges
        while len(edge_pos) > 0:
            pos4d = edge_pos.pop()
            new_edge = deque(pos4d)
            left_theta = utils.angle_norm(pos4d.theta - 90)
            dx, dz = utils.orientation_bindings[left_theta]
            dx *= self.grid_size
            dz *= self.grid_size
            neighbor = pos4d + (dx, dz)
            while neighbor in edge_pos:
                new_edge.appendleft(neighbor)
                edge_pos.remove(neighbor)
                neighbor = neighbor + (dx, dz)
            right_theta = utils.angle_norm(pos4d.theta + 90)
            dx, dz = utils.orientation_bindings[right_theta]
            dx *= self.grid_size
            dz *= self.grid_size
            neighbor = pos4d + (dx, dz)
            while neighbor in edge_pos:
                new_edge.append(neighbor)
                edge_pos.remove(neighbor)
                neighbor = neighbor + (dx, dz)
            edges.append(new_edge)
        interval = int(np.ceil(0.5 / self.grid_size))
        edges = [list(x) for x in edges if len(x) > interval]

        # generate location on edges
        receptacles = {
            "CounterTop": 0,
            "DiningTable": 0,
            "Shelf": 0,
            "SideTable": 0,
            "SinkBasin": 0,
            "StoveBurner": 0,
        }
        for edge in edges:
            for pos4d in edge[interval // 2 :: interval]:
                event = self.controller.step(
                    action="Teleport",
                    position=dict(x=pos4d.x, y=self.agent_["position"]["y"], z=pos4d.z),
                    rotation=dict(x=0, y=pos4d.theta, z=0),
                    horizon=15,
                    standing=True,
                )
                for key, mask in event.instance_masks.items():
                    for receptacle_type, count in receptacles.items():
                        if key.startswith(receptacle_type):
                            if mask.mean() > 0.05:
                                locations[
                                    "{}_{}".format(receptacle_type, count)
                                ] = pos4d
        self.locations = locations
        self.current_location = list(self.locations.keys())[0]
        pos4d = self.locations[self.current_location]
        self.event = self.controller.step(
            action="Teleport",
            position=dict(x=pos4d.x, y=self.agent_["position"]["y"], z=pos4d.z),
            rotation=dict(x=0, y=pos4d.theta, z=0),
            horizon=pos4d.y,
            standing=True,
        )
        self.navigation_cache = {}

    def _is_ignored(self, obj_info: Dict[str, Any]) -> bool:

        # ignore some objects to avoid excessive exploration
        return (
            (
                obj_info["objectType"] in {"Cabinet", "Drawer"}
                and obj_info["name"] not in self.locations.keys()
            )
            or obj_info["objectType"] in {"DishSponge", "SaltShaker", "PepperShaker"}
            or (
                (obj_info["objectType"] + "Sliced")
                in {info["objectType"] for info in self.event.metadata["objects"]}
            )
        )

    def _is_strictly_visible(self, obj_info: Dict[str, Any]) -> bool:
        return (
            (
                obj_info["visible"]
                or self.event.instance_masks.get(
                    obj_info["objectId"], np.zeros((1,))
                ).mean()
                > 0.1
            )
            and not self._is_ignored(obj_info)
            and obj_info["objectType"] != "Floor"
            and ("Slice" not in obj_info["name"] or "Slice" in obj_info["objectType"])
        )

    def _is_visible(self, obj_info: Dict[str, Any]) -> bool:

        if self._get_location(obj_info) == self.current_location:
            location_object = self._get_object_by_name(self.current_location)
            return (
                location_object is None
                or not location_object["openable"]
                or location_object["isOpen"]
                or obj_info["name"] == self.current_location
            )
        else:
            return self._is_strictly_visible(obj_info)

    def _set_object_pose(self, positions: dict, rotations: dict):

        objects = [
            dict(
                objectName=x["name"],
                position=positions.get(x["objectId"], x["position"]),
                rotation=rotations.get(x["objectId"], x["rotation"]),
            )
            for x in self.objects_
            if x["moveable"] or x["pickupable"]
        ]
        self.api_step(action="SetObjectPoses", objectPoses=objects, wait=False)
        self.api_step(action="Done", wait=False)

    def _get_object_pose(self) -> List[Dict[str, Any]]:

        return [
            dict(
                objectName=x["name"],
                position=x["position"],
                rotation=x["rotation"],
            )
            for x in self.objects_
            if x["moveable"] or x["pickupable"]
        ]

    def _update_screen(self, event: Event):

        rgb = event.frame.transpose((1, 0, 2))
        bird = event.third_party_camera_frames[0].transpose((1, 0, 2))
        self.screen.blit(pygame.surfarray.make_surface(rgb), self.rgb_top_left)
        self.screen.blit(pygame.surfarray.make_surface(bird), self.bird_top_left)

        obj_frame = np.zeros_like(event.instance_segmentation_frame)
        for obj_info in event.metadata["objects"]:
            if obj_info[
                "objectId"
            ] in event.instance_masks.keys() and self._is_strictly_visible(obj_info):
                obj_frame[
                    event.instance_masks[obj_info["objectId"]], :
                ] = event.object_id_to_color[obj_info["objectId"]]
        obj_frame = obj_frame.transpose(1, 0, 2)
        self.screen.blit(pygame.surfarray.make_surface(obj_frame), self.obj_top_left)

        pygame.display.flip()

    def _all_countertop_names(self) -> List[str]:
        return [
            obj_info["name"]
            for obj_info in self.objects_
            if obj_info["objectType"] == "CounterTop"
        ]

    def _object_name_to_id(self, object_name: str) -> str:

        for obj_info in self.objects_:
            if obj_info["name"] == object_name:
                return obj_info["objectId"]
        if object_name.startswith("CounterTop"):
            countertops = [
                obj_info
                for obj_info in self.objects_visible_
                if obj_info["objectType"] == "CounterTop"
            ]
            if countertops != []:
                return countertops[0]["objectId"]
        logging.warning(
            "object name {} not found! returning empty string".format(object_name)
        )
        return ""

    def _get_object_by_name(self, object_name: str) -> Dict[str, Any]:

        for obj_info in self.objects_:
            if obj_info["name"] == object_name:
                return obj_info
        logging.warning("object name {} not found".format(object_name))

    def _get_object_by_id(self, object_id: str) -> Dict[str, Any]:

        return self.event.get_object(object_id)

    def _sample_openness(self, p_open: float = 0.2):

        self._close_all_containers()
        for obj_info in self.objects_visible_:
            if obj_info["openable"]:
                if random.random() < p_open:
                    self.api_step(action="OpenObject", objectId=obj_info["objectId"])

    def _sample_objects_on_countertop(self, n: int):

        pickable_objects = [
            obj_info["objectId"]
            for obj_info in self.objects_
            if obj_info["pickupable"]
            and self._get_location(obj_info) is not None
            and self._get_location(obj_info).startswith("CounterTop")
        ]
        chosen = random.sample(pickable_objects, n)
        for obj_id in pickable_objects:
            if obj_id not in chosen:
                self.api_step(action="DisableObject", objectId=obj_id)

    def _close_all_containers(self):

        for obj_info in self.objects_:
            if obj_info["openable"] and obj_info["isOpen"]:
                self.api_step(action="CloseObject", objectId=obj_info["objectId"])

    def _get_closest_countertop(self) -> str:

        return min(
            [
                location
                for location in self.locations.keys()
                if location.startswith("CounterTop")
            ],
            key=lambda x: self.locations[self.current_location] - self.locations[x],
        )

    def _get_location(self, obj_info: Dict[str, Any]) -> Optional[str]:

        support = self._get_support(obj_info)
        if support is not None:
            if support in self.locations.keys() or support == gripper_name:
                return support
            support_location = self._get_location(self._get_object_by_name(support))
            if support_location is not None:
                return support_location
        else:
            return None

    def _get_support(self, obj_info: Dict[str, Any]) -> Optional[str]:

        if (
            self.object_in_hand is not None
            and obj_info["objectId"] == self.object_in_hand["objectId"]
        ):
            return gripper_name

        if obj_info["objectType"] in {
            "CounterTop",
            "Drawer",
            "Fridge",
            "Cabinet",
            "SinkBasin",
        }:
            return None

        res = None
        if obj_info["name"] in self.stacking_info:
            res = self.stacking_info[obj_info["name"]]
        elif obj_info["parentReceptacles"] is not None:
            recep_id = obj_info["parentReceptacles"][-1]
            if not recep_id.startswith("Floor"):
                if recep_id.startswith("CounterTop"):
                    res = self._get_closest_countertop()
                else:
                    res = self._get_object_by_id(recep_id)["name"]

        if res is not None and "Sink" in res and res not in self.locations.keys():
            res = {x for x in self.locations if "Sink" in x}.pop()
        if res is not None and "Stove" in res and res not in self.locations.keys():
            res = {x for x in self.locations if "Stove" in x}.pop()

        return res

    def _get_object_name_from_type(self, object_type: str) -> Optional[str]:

        for obj_info in self.objects_visible_:
            if obj_info["objectType"].lower() == object_type.lower():
                return obj_info["name"]
        return None

    # === api perception === #

    @property
    def robot_status(self) -> List[Tuple[str, str, str]]:

        object_in_hand = self.object_in_hand
        return [
            ("Robot", "at", self.current_location),
            (gripper_name, "is_empty", "true")
            if object_in_hand is None
            else (
                self._get_object_by_id(object_in_hand["objectId"])["name"],
                "in",
                gripper_name,
            ),
        ]

    @property
    def object_status(self) -> List[Tuple[str, str, str]]:

        res = []
        for obj_info in self.objects_visible_:
            support = self._get_support(obj_info)
            if support is not None:
                res.append((obj_info["name"], "in/on", support))
            if obj_info["openable"]:
                if obj_info["isOpen"]:
                    res.append((obj_info["name"], "is_open", "true"))
                    if (
                        self.current_location == obj_info["name"]
                        and obj_info["receptacle"]
                        and obj_info["receptacleObjectIds"] == []
                    ):
                        res.append((obj_info["name"], "is_empty", "true"))
                else:
                    res.append((obj_info["name"], "is_closed", "true"))

        return res

    def get_status(self) -> List[Tuple[str, str, str]]:

        return self.robot_status + self.object_status

    def get_available_actions(self) -> List[str]:

        res = []

        for location in self.locations.keys():
            if location != self.current_location:
                res.append("move to {}".format(location))

        object_in_hand = self.object_in_hand
        for obj_info in self.objects_visible_:
            if obj_info["openable"] and (
                obj_info["name"] == self.current_location
                or obj_info["name"] not in self.locations.keys()
            ):
                res.append(
                    "{} {}".format(
                        "close" if obj_info["isOpen"] else "open", obj_info["name"]
                    )
                )
            if obj_info["toggleable"]:
                if obj_info["isToggled"]:
                    res.append("toggle off {}".format(obj_info["name"]))
                else:
                    res.append("toggle on {}".format(obj_info["name"]))
            if (
                object_in_hand is None
                and obj_info["pickupable"]
                and self.current_location == self._get_location(obj_info)
            ):
                res.append("pick up {}".format(obj_info["name"]))
            if (
                object_in_hand is not None
                and "Knife" in object_in_hand["objectType"]
                and obj_info["sliceable"]
            ):
                res.append("slice {}".format(obj_info["name"]))
            if (
                object_in_hand is not None
                and (obj_info["receptacle"] or "Slice" in obj_info["objectType"])
                and obj_info["objectId"] != object_in_hand["objectId"]
                and (not obj_info["openable"] or obj_info["isOpen"])
            ):
                recep_name = obj_info["name"]
                if recep_name.startswith("CounterTop"):
                    recep_name = self._get_closest_countertop()
                res.append(
                    "put {} on {}".format(
                        self._get_object_by_id(object_in_hand["objectId"])["name"],
                        recep_name,
                    )
                )

        return res

    def get_distance(self, loc1: str, loc2: str) -> float:

        return self.locations[loc1] - self.locations[loc2]

    # === api actions === #

    def parse_actions(self, action_str: str):

        tokens = action_str.split(" ")
        name = tokens[0]
        arg = tokens[-1]
        renaming = {
            "move": "move_to",
            "pick": "pick_up",
            "put": "put_down",
            "place": "put_down",
        }
        if name in renaming:
            name = renaming[name]
        if name == "toggle":
            name = name + "_" + tokens[1]

        getattr(self, name)(arg)

    def move_to(self, location: str, forbid_same_location: bool = True):

        if location not in self.locations.keys():
            logging.warning(
                pformat(
                    "location {} not found, available locations: \n{}".format(
                        location, self.locations
                    ),
                    indent=2,
                )
            )
            raise AffordanceError("location {} not found".format(location))
            return

        if location == self.current_location:
            logging.warning("robot already at location {}".format(location))
            if forbid_same_location:
                raise AffordanceError("robot already at location {}")
            return

        self.current_location = location
        logging.debug("moving to {}".format(location))
        self._teleport_pos4d(self.locations[self.current_location])

    def look_at(self, object_name: str):
        pass

    def pick_up(self, object_name: str):

        if object_name in self.objects_visible:
            self.api_step(
                action="PickupObject",
                objectId=self._object_name_to_id(object_name),
            )
            self.api_step(action="Done")
            if object_name in self.stacking_info.keys():
                self.stacking_info.pop(object_name)
            sleep(self.long_interval)
        else:
            object_name = self._get_object_name_from_type(object_name)
            if object_name is None:
                logging.warning("{} not found in scene".format(object_name))
            else:
                self.pick_up(object_name)

    def open(self, object_name: str):

        if object_name in self.objects_visible:
            # self.look_at_obj(object_name)
            self.api_step(
                action="OpenObject",
                objectId=self._object_name_to_id(object_name),
                forceAction=True,
            )
            self.api_step(action="Done")
            sleep(self.long_interval)
        else:
            logging.warning("{} not found in scene".format(object_name))

    def close(self, object_name: str):

        if object_name in self.objects_visible:
            self.api_step(
                action="CloseObject",
                objectId=self._object_name_to_id(object_name),
                forceAction=True,
            )
            self.api_step(action="Done")
            sleep(self.long_interval)
        else:
            logging.warning("{} not found in scene".format(object_name))

    def drop(self):

        if self.event.metadata["inventoryObjects"] != []:
            # self.api_step(action="DropHandObject", forceAction=True)
            self.api_step(action="RotateLeft", degrees=30)
            self.api_step(action="LookUp", degrees=10)
            self.api_step(action="ThrowObject", moveMagnitude=80, forceAction=True)
            self.api_step(action="RotateRight", degrees=30)
            self.api_step(action="LookDown", degrees=10)
        else:
            logging.warning("attempting to drop object when no object in hand")

    def slice(self, object_name: str):

        if object_name in self.objects_visible:
            obj_type = self._get_object_by_name(object_name)["objectType"]
            location = self.stacking_info.get(object_name, None)
            self.api_step(
                action="SliceObject", objectId=self._object_name_to_id(object_name)
            )
            if location is not None:
                for obj_info in self.objects_:
                    if obj_info["objectType"] == obj_type + "Sliced":
                        self.stacking_info[obj_info["name"]] = location
            self.api_step(action="Done")
            sleep(self.long_interval)
        else:
            object_name = self._get_object_name_from_type(object_name)
            if object_name is None:
                logging.warning("{} not found in scene".format(object_name))
            else:
                self.slice(object_name)

    def toggle_obj(self, object_name: str, state: bool):

        if object_name in self.objects_visible:
            self.api_step(
                action="ToggleObject" + "On" if state else "Off",
                objectId=self._object_name_to_id(object_name),
            )
            self.api_step(action="Done")
            sleep(self.long_interval)
        else:
            logging.warning("{} not found in scene".format(object_name))

    def toggle_on(self, object_name: str):

        return self.toggle_obj(object_name, True)

    def toggle_off(self, object_name: str):

        return self.toggle_obj(object_name, False)

    def put_down(self, recep_name: str):
        """Manually handle putting down sliced objects"""

        recep_id = self._object_name_to_id(recep_name)
        try:
            object_in_hand = self.event.metadata["inventoryObjects"][0]["objectId"]
            object_in_hand_name = self._get_object_by_id(object_in_hand)["name"]
        except IndexError:
            logging.warning("agent has no object in hand to be put down")
            raise AffordanceError("agent has no object in hand to be put down")
        else:
            success = False
            if recep_name.startswith("CounterTop"):
                locations = [self.current_location] + list(self.locations.keys())
                for location in locations:
                    if location.startswith("CounterTop"):
                        if location != self.current_location:
                            self.move_to(location)
                        self.api_step(
                            action="PutObject",
                            objectId=self._object_name_to_id(location),
                            raise_error=False,
                        )
                        if self.event.metadata["lastActionSuccess"]:
                            self.stacking_info[object_in_hand_name] = location
                            success = True
                            break

            else:
                event = self.api_step(
                    action="PutObject",
                    objectId=recep_id,
                    raise_error=False,
                )
                if event.metadata["lastActionSuccess"]:
                    success = True
                    self.stacking_info[object_in_hand_name] = recep_name
                    if "Slice" in object_in_hand and "Toaster" not in recep_name:
                        current_object = event.get_object(object_in_hand)
                        target_object = event.get_object(recep_id)
                        target_bbox = target_object["axisAlignedBoundingBox"]
                        rotation = current_object["rotation"]
                        position = current_object["position"]
                        rotation["x"] = 90
                        position["y"] = (
                            max(pt[1] for pt in target_bbox["cornerPoints"])
                            + target_bbox["size"]["y"] / 2
                        )
                        self._set_object_pose(
                            {object_in_hand: position},
                            {object_in_hand: rotation},
                        )
                elif "Slice" in object_in_hand:
                    success = True
                    self.stacking_info[object_in_hand_name] = recep_name
                    current_object = event.get_object(object_in_hand)
                    target_object = event.get_object(recep_id)
                    target_bbox = target_object["axisAlignedBoundingBox"]
                    rotation = current_object["rotation"]
                    position = target_bbox["center"]
                    rotation["x"] = 90
                    position["y"] = (
                        max(pt[1] for pt in target_bbox["cornerPoints"])
                        + target_bbox["size"]["y"]
                    )
                    self._set_object_pose(
                        {object_in_hand: position},
                        {object_in_hand: dict(x=90, y=0, z=0)},
                    )
                    self.api_step(action="DropHandObject", forceAction=True, wait=False)
                else:
                    logging.warning(
                        "putting {} onto {} failed, trying again".format(
                            object_in_hand, recep_id
                        )
                    )
                    self.api_step(
                        action="PutObject",
                        objectId=recep_id,
                        forceAction=True,
                        raise_error=False,
                    )
                    if self.event.metadata["lastActionSuccess"]:
                        logging.warning("succeeded with forceAction")
                        success = True
                        self.stacking_info[object_in_hand_name] = recep_name
                    else:
                        logging.warning("failed with forceAction")
                        if recep_name.startswith("Cabinet"):
                            for obj_info in self.objects_visible_:
                                if (
                                    not success
                                    and obj_info["objectType"] == "Cabinet"
                                    and obj_info["name"] != recep_name
                                ):
                                    self.api_step(
                                        action="PutObject",
                                        objectId=obj_info["objectId"],
                                        forceAction=True,
                                        raise_error=False,
                                    )
                                    success = self.event.metadata["lastActionSuccess"]
            if not success:
                raise AffordanceError(
                    "putting {} onto {} failed with forceAction".format(
                        object_in_hand, recep_id
                    )
                )

        self.api_step(action="Done")
        sleep(self.long_interval)
