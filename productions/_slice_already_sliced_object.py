# flake8: noqa
from production import *

production_name = "SliceAlreadySlicedObject"


class SliceAlreadySlicedObject(Production):
    english_description = "IF the current task is to slice a/an <sliceable> AND the <sliceable> is already sliced AND the robot's gripper is holding a knife THEN choose Special action: 'done'."

    target_task: str = "slice a/an <sliceable>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        # Parse the target object from the current task
        if not current_task.startswith("slice"):
            return False, "The current task is not to slice an object"
        target_object_name = current_task.split(" ")[2]
        target_object = object_knowledge[target_object_name]
        if target_object is None:
            return False, f"No knowledge about the target object {target_object_name}"
        setattr(self, "target_object", target_object)

        # Check if the target object is already sliced
        if not target_object.sliced:
            return False, f"The target object {target_object_name} is not sliced"

        # Check if the robot's gripper is holding a knife
        knife = spatial_knowledge["RobotGripper"].hosting[0] if spatial_knowledge["RobotGripper"].hosting else None
        if knife is None or not name_equal(knife.object_type, "knife"):
            return False, "The robot's gripper is not holding a knife"
        setattr(self, "knife", knife)

        return True, ""

    def apply(self) -> str:
        return "special action: done"
