import logging
import os
import pdb
import random
from collections import UserList, deque
from typing import Dict, List, NamedTuple, Optional

import cv2

from env.env import Env as EnvBase
from utils import AffordanceError, gripper_name, gripper_type, name_split

random.seed(0)


def name_equal(name1: Optional[str], name2: Optional[str]) -> bool:

    if name1 is None or name2 is None or name1 == "" or name2 == "":
        return False
    split1 = name_split(name1)
    split2 = name_split(name2)
    return (
        split1.issuperset(split2)
        or split2.issuperset(split1)
        or (
            (name1.lower() == "countertop" or name2.lower() == "countertop")
            and name1.lower().startswith("countertop")
            and name2.lower().startswith("countertop")
        )
    )


class NameCompatibleList(UserList):
    def __init__(self, contents: Optional[list] = []):
        self.data = contents

    def __contains__(self, key) -> bool:
        for item in self.data:
            if name_equal(item, key):
                return True

    def index(self, key: str):
        for i, item in self.data:
            if name_equal(item, key):
                return i
        raise ValueError("{} not in list".format(key))


class LocationHistory:
    def __init__(self):
        self.history = deque()

    def __repr__(self) -> str:

        if len(self.history) == 0:
            return " None"
        else:
            return "\n".join(
                " * Robot at {} from time {} to {}".format(*entry)
                for entry in self.history
            )

    def __contains__(self, location: str) -> bool:

        for loc, _, _ in self.history:
            if loc == location:
                return True
        return False

    def get_current_location(self) -> str:

        if len(self.history) == 0:
            logging.error("trying to get current location when the history is empty")
            return ""
        else:
            return self.history[-1][0]

    def get_current_time(self) -> int:

        if len(self.history) == 0:
            return 0
        else:
            return self.history[-1][-1]

    def add_location(self, location: str, timestep: int):

        if len(self.history) == 0 or self.get_current_location() != location:
            self.history.append([location, self.get_current_time(), timestep])
        else:
            self.history[-1][-1] += 1


class ActionHistory:
    def __init__(self):
        self.history = []
        self.clock = 1

    def __repr__(self) -> str:

        return "\n".join(
            " * (time {}) {} (justification: {}))".format(*entry)
            for entry in self.history
        )

    def get_recent(self, k: int = 5) -> str:

        if self.history == []:
            return " None"
        else:
            entries = self.history[-k:] if len(self.history) > k else self.history
            return "\n".join(self.format_entry(entry) for entry in entries)

    def format_entry(self, entry) -> str:

        res = " * (time {}) {}".format(*entry[:2])
        if entry[2] is not None:
            res += " (purpose: {})".format(entry[2])
        if entry[3] is not None:
            res += " Result: {}".format(entry[3])
        return res

    def add_action(
        self,
        action_str: str,
        justification: Optional[str] = None,
        result: Optional[str] = None,
        time: Optional[int] = None,
    ):

        time = self.clock if time is None else time
        self.history.append((time, action_str, justification, result))
        self.clock = time + 1


class Object(NamedTuple):
    object_name: str
    object_type: str
    location: str
    supporting: str
    openable: bool = False
    opened: bool = False
    pickable: bool = False
    toggleable: bool = False
    toggled: bool = False
    cookable: bool = False
    cooked: bool = False
    dirtyable: bool = False
    dirty: bool = False
    sliced: bool = False

    def __hash__(self) -> int:
        return hash(self.object_name)

    def __repr__(self) -> str:

        res = "{}({})".format(self.object_name, self.object_type)
        if self.location is not None:
            res += " at {}".format(self.location)
        if self.openable or self.toggleable or self.cookable:
            res += ":"
        if self.supporting != self.location:
            res += " on {},".format(self.supporting)
        if self.openable:
            res += " opened," if self.opened else " closed,"
        if self.toggleable:
            res += " turned on," if self.toggled else "turned off,"
        if self.cookable:
            res += " cooked," if self.cooked else " uncooked,"
        if self.dirtyable:
            res += " dirty," if self.dirty else " clean,"
        if self.sliced:
            res += " sliced,"
        return res

    def __eq__(self, other):

        if isinstance(other, Object):
            return self.object_name == other.object_name
        elif isinstance(other, str):
            if name_equal(self.object_name, other):
                return True
            elif other.lower().startswith("cooked"):
                return other.lower()[6:] == self.object_type.lower() and self.cooked
            return False
        return False

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return self.object_name < other.object_name

    @property
    def short_name(self) -> str:

        if self.supporting == self.location:
            return "{}({})".format(self.object_name, self.object_type)
        else:
            return "{}({}) in/on {}".format(
                self.object_name, self.object_type, self.supporting
            )


class Receptacle:
    def __init__(
        self, object_name: str, object_type: str, hosting: Optional[List[Object]] = None
    ):

        self.object_name = object_name
        self.object_type = object_type
        self.hosting = hosting
        self.visited = False

    def __hash__(self):
        return hash(self.object_name)

    def __repr__(self) -> str:

        if self.hosting is None:
            return (
                "{}({}) is unexplored: potentially has other objects on/in it".format(
                    self.object_name, self.object_type
                )
            )
        elif self.hosting == []:
            return "{}({}) has been explored: it is empty".format(
                self.object_name, self.object_type
            )
        else:
            object_str = ", ".join(obj.short_name for obj in self.hosting)
            if not self.visited:
                return (
                    "{}({}) is partially explored: it has {}, "
                    "and potentially other objects".format(
                        self.object_name, self.object_type, object_str
                    )
                )
            else:
                return (
                    "{}({}) has been fully explored: it has {}, "
                    "and nothing else".format(
                        self.object_name, self.object_type, object_str
                    )
                )

    def __contains__(self, key: str):
        return self.hosting is not None and key in self.hosting

    def __eq__(self, other):

        return (
            (isinstance(other, str) and name_equal(other, self.object_name))
            # or (isinstance(other, str) and name_equal(other, self.object_type))
            or (isinstance(other, Receptacle) and other.object_name == self.object_name)
        )

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other: "Receptacle"):

        return self.object_name == gripper_name or self.object_name < other.object_name

    def clear(self):

        self.hosting = []

    def add(self, obj: Object):

        if self.hosting is None:
            self.hosting = [obj]
        elif obj not in self.hosting:
            self.hosting.append(obj)

    def remove(self, obj: Object):

        if self.hosting is not None:
            self.hosting = [x for x in self.hosting if not x == obj]

    def get_hosting(self) -> Optional[List[Object]]:
        return self.hosting

    def visit(self):
        self.visited = True
        if self.hosting is None:
            self.hosting = []


class Gripper(Receptacle):
    def __init__(self):
        self.object_name = gripper_name
        self.object_type = gripper_type
        self.hosting = []
        self.visited = True

    def __repr__(self) -> str:

        if self.hosting == []:
            return "{}({}) is empty".format(self.object_name, self.object_type)
        else:
            object_str = ", ".join(obj.short_name for obj in self.hosting)
            return "{}({}) has {}, and nothing else".format(
                self.object_name, self.object_type, object_str
            )


class SpatialKnowledge:
    def __init__(
        self, current_location: str, location_matrix: Dict[str, Dict[str, float]]
    ):

        self.location_matrix = location_matrix
        self.locations = {
            loc: Receptacle(
                loc,
                "CounterTop"
                if loc.lower().startswith("countertop")
                else loc.split("_")[0],
            )
            for loc in location_matrix.keys()
        }
        self.locations[gripper_name] = Gripper()
        self.current_location = current_location

    def __contains__(self, key) -> bool:
        return self[key] is not None

    def __hash__(self) -> int:

        return abs(
            sum(
                (hash(str(x)) for x in self.locations.values()),
                hash(self.current_location),
            )
        )

    def __repr__(self):

        dist_and_recep = sorted(
            (self.get_distance(name), loc) for name, loc in self.locations.items()
        )
        return "\n".join(
            " * ({:.1f} meters away) {}".format(*entry) for entry in dist_and_recep
        )

    def __getitem__(self, key: str) -> Optional[Receptacle]:

        res = self.locations.get(key)
        if res is None:
            min_dist = None
            for name, recep in sorted(
                self.locations.items(), key=lambda x: self.get_distance(x[0])
            ):
                if recep == key:
                    new_dist = self.get_distance(name)
                    if min_dist is None or new_dist < min_dist:
                        res = recep
                        min_dist = new_dist
        return res

    def update(self, current_location: str, observations: Dict[str, Object]):

        self.current_location = current_location
        # if current_location.startswith("CounterTop"):
        #     self.locations[current_location].visit()
        if (
            current_location.lower().startswith("countertop")
            or current_location.lower().startswith("sink")
            or current_location not in observations.keys()
            or observations[current_location].opened
        ):
            self.locations[current_location].clear()
            if not self.locations[current_location].visited:
                self.locations[current_location].visit()
        self.locations[gripper_name].clear()
        for obj in observations.values():
            if obj.object_name == current_location and (obj.opened or not obj.openable):
                self.locations[current_location].visit()
            if obj.location is None:
                logging.debug("{} has location None".format(obj.object_name))
            elif obj.location != obj.object_name:
                logging.debug("adding {} to {}".format(obj.object_name, obj.location))
                self.remove_object(obj)
                self.locations[obj.location].add(obj)
            else:
                logging.debug("ignored {} in spatial knowledge".format(obj.object_name))

    def get_distance(self, location: str) -> float:

        if location in [gripper_name, gripper_type]:
            return 0
        elif location in self.locations.keys():
            return self.location_matrix[self.current_location][location]
        else:
            for recep_name, recep in self.locations.items():
                if location in recep:
                    return self.get_distance(recep_name)
            return float("inf")

    def get_all_receptacles(self) -> List[Receptacle]:

        return self.locations

    def remove_object(self, obj: Object):

        for receptacle in self.locations.values():
            receptacle.remove(obj)


class ObjectKnowledge:
    def __init__(self, spatial_knowledge: SpatialKnowledge):
        self.objects: Dict[str, Object] = {}
        self.spatial_knowledge = spatial_knowledge

    def __repr__(self):

        return "\n".join(" * {}".format(obj) for obj in sorted(self.objects.values()))

    def __hash__(self) -> int:

        return abs(sum(hash(str(x)) for x in self.objects.values()))

    def __getitem__(self, key):

        if key in self.objects.keys():
            return self.objects[key]
        if isinstance(key, Receptacle):
            return self.objects[key.object_name]
        for obj in sorted(
            self.objects.values(),
            key=lambda obj: self.spatial_knowledge.get_distance(
                self._get_location(obj)
            ),
        ):
            if obj == key:
                return obj

    def __contains__(self, item):

        return self[item] is not None

    def _get_location(self, obj: Object) -> str:

        if obj.object_name in self.spatial_knowledge.locations.keys():
            return obj.object_name
        else:
            return obj.location

    def update(self, observations: Dict[str, Object]):

        for obj in observations.values():
            if obj.object_type != "CounterTop":
                self.objects[obj.object_name] = obj
            if obj.sliced:
                original_type = obj.object_type.replace("Sliced", "")
                to_remove = set()
                for name, obj in self.objects.items():
                    if obj.object_type == original_type:
                        to_remove.add(name)
                for name in to_remove:
                    self.objects.pop(name)


class Env(EnvBase):
    def _save_screenshot(self):
        if hasattr(self, "log_dir"):
            cv2.imwrite(
                os.path.join(self.log_dir, "{:05d}.jpg".format(self.count)),
                self.event.cv2img,
            )

    def set_log_dir(self, log_dir: str):
        self.log_dir = log_dir
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.count = 0
        self._save_screenshot()

    def get_observation(self) -> Dict[str, Object]:

        # supports = {
        #     obj_info["name"]: (self._get_support(obj_info), obj_info)
        #     for obj_info in self.objects_visible_
        # }
        # visited = set()
        # obj_dict = {}

        # for obj_name, (support_name, obj_info) in supports.items():
        #     if obj_name not in visited:
        #         visited.add(obj_name)
        #         stack = deque()
        #         while support_name in supports.keys():
        #             visited.add(support_name)
        #             stack.append(obj_info)
        #             new_support_name, obj_info = supports[support_name]
        #             if new_support_name is not None:
        #                 support_name = new_support_name
        #             else:
        #                 break
        #         stack.append(obj_info)
        #         for obj_info in stack:
        #             obj_dict[obj_info["name"]] = Object(
        #                 obj_info["name"],
        #                 obj_info["objectType"],
        #                 support_name,
        #                 supports[obj_info["name"]][0],
        #                 obj_info["openable"],
        #                 obj_info["isOpen"],
        #                 obj_info["pickupable"],
        #                 obj_info["toggleable"],
        #                 obj_info["isToggled"],
        #                 obj_info["cookable"],
        #                 obj_info["isCooked"],
        #                 obj_info["dirtyable"],
        #                 obj_info["dirtyable"] and obj_info["isDirty"],
        #                 "Slice" in obj_info["name"],
        #             )

        obj_dict = {}
        for obj_info in self.objects_:
            obj_location = self._get_location(obj_info)
            if (self._is_visible(obj_info)) and (
                not name_equal(self._get_location(obj_info), "CounterTop")
                or obj_info["pickupable"]
            ):
                obj_dict[obj_info["name"]] = Object(
                    obj_info["name"],
                    obj_info["objectType"],
                    obj_location,
                    self._get_support(obj_info),
                    obj_info["openable"],
                    obj_info["isOpen"],
                    obj_info["pickupable"],
                    obj_info["toggleable"],
                    obj_info["isToggled"],
                    obj_info["cookable"],
                    obj_info["isCooked"],
                    obj_info["dirtyable"],
                    obj_info["dirtyable"] and obj_info["isDirty"],
                    "Slice" in obj_info["name"],
                )

        return obj_dict

    def get_actions(self) -> List[str]:
        action_list = self.get_available_actions()
        action_list = [
            "motor action: " + x for x in action_list if not x.startswith("move")
        ]
        action_list.append("motor action: move to <receptacle>")
        return action_list

    def perform_action(self, action_str: str):

        tokens = action_str.strip().split(" ")
        action_name = tokens[0].lower()
        obj_name = tokens[-1].lower()
        for candidate_str in self.get_available_actions():
            candidate_str_lower = candidate_str.lower()
            if action_name in candidate_str_lower and obj_name in candidate_str_lower:
                matched = candidate_str
                logging.debug(
                    "converting '{}' to '{}'".format(action_str, candidate_str)
                )
                break
        else:
            if (
                action_name == "move"
                and obj_name.startswith("countertop")
                and obj_name != "countertop"
            ):
                return self.perform_action("{} CounterTop".format(action_name))
            raise AffordanceError("{} not parsable".format(action_str))
        try:
            res = self.parse_actions(matched)
        except Exception as e:
            res = None
            if isinstance(e, AffordanceError):
                raise e
            else:
                logging.error(
                    "env throws {}: {} when trying to execute {}".format(
                        type(e), e, action_str
                    )
                )
                pdb.set_trace()
        self.count += 1
        self._save_screenshot()
        return res

    def get_objects(self):

        return {
            "moveables": [
                dict(
                    objectName=obj_info["name"],
                    position=obj_info["position"],
                    rotation=obj_info["rotation"],
                )
                for obj_info in self.objects_
                if obj_info["moveable"] or obj_info["pickupable"]
            ],
            "openables": {
                obj_info["objectId"]: obj_info["isOpen"]
                for obj_info in self.objects_
                if obj_info["openable"]
            },
        }

    def set_objects(self, objects):

        self.api_step(
            action="SetObjectPoses", objectPoses=objects["moveables"], wait=False
        )
        for object_id, is_open in objects["openables"].items():
            if is_open:
                self.api_step(action="OpenObject", objectId=object_id, forceAction=True)
            else:
                self.api_step(
                    action="CloseObject", objectId=object_id, forceAction=True
                )

        self.api_step(action="Done", wait=False)

    def sample_location(self):

        try:
            self.move_to(random.choice(list(self.locations.keys())))
        except AffordanceError:
            pass

    def set_object_pose(self, object_pose_file: Optional[str] = None):
        EnvBase.object_pose_file = object_pose_file
