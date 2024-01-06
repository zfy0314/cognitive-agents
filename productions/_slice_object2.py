# flake8: noqa
from production import *

production_name = "SliceObject2"


class SliceObject2(Production):
    english_description = "IF the current task is to slice a/an <sliceable> AND the robot's gripper is holding a knife AND the <sliceable> is located on the countertop in front of the robot THEN choose motor action: slice <sliceable>."

    target_task: str = "slice a/an <sliceable>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        match = re.match(r"slice a/an (\w+)", current_task)
        if not match:
            return False, "The current task is not to slice an object"
        target_object = match.group(1)

        if not spatial_knowledge["RobotGripper"].hosting or not name_equal(spatial_knowledge["RobotGripper"].hosting[0].object_type, "knife"):
            return False, "The robot's gripper is not holding a knife"

        target_location = None
        for receptacle in spatial_knowledge.locations.values():
            if target_object in receptacle:
                target_location = receptacle.object_name
                break
        if not target_location or not name_equal(target_location, current_location):
            return False, f"The {target_object} is not located on the countertop in front of the robot"

        setattr(self, "target_object", target_object)
        return True, ""

    def apply(self) -> str:
        return f"motor action: slice {self.target_object}"
