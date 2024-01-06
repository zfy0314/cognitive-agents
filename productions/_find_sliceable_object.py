# flake8: noqa
from production import *

production_name = "FindSliceableObject"


class FindSliceableObject(Production):
    english_description = "IF the current task is to slice a/an <sliceable_object> AND the robot's gripper is empty AND the robot has not found a/an <sliceable_object> yet THEN choose 'attend to subtask: find a/an <sliceable_object>'."

    target_task: str = "slice a/an <sliceable>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:

        # Parse the sliceable object from the current task
        sliceable_object = re.search(r'slice a/an (\w+)', current_task)
        if sliceable_object is None:
            return False, "The current task is not to slice a sliceable object"
        sliceable_object = sliceable_object.group(1)

        # Get the robot's gripper
        gripper = spatial_knowledge["RobotGripper"]

        # Check if the robot's gripper is empty
        if gripper.hosting != []:
            return False, "The robot's gripper is not empty"

        # Check if the robot has found the sliceable object
        found_sliceable_object = None
        for obj in object_knowledge.objects.values():
            if name_equal(obj.object_type, sliceable_object):
                found_sliceable_object = obj
                break
        if found_sliceable_object is not None:
            return False, f"The robot has already found a/an {sliceable_object}"

        # Set variable bindings
        setattr(self, "sliceable_object", sliceable_object)

        return True, ""

    def apply(self) -> str:
        return f"attend to subtask: find a/an {self.sliceable_object}"
