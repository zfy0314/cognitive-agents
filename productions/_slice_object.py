# flake8: noqa
from production import *

production_name = "SliceObject"


class SliceObject(Production):
    english_description = "IF the current task is to slice a/an <object> AND the robot is holding the <object> in its gripper AND the robot is not at a suitable place for slicing AND there is a countertop available THEN choose motor action: put <object> on <countertop>."

    target_task: str = "slice a/an <sliceable>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        match = re.match(r"slice a/an (.+)", current_task)
        if not match:
            return False, "The current task is not to slice an object"
        target_object = match.group(1)

        if not spatial_knowledge["RobotGripper"].hosting or not name_equal(spatial_knowledge["RobotGripper"].hosting[0].object_type, target_object):
            return False, "The robot is not holding the target object in its gripper"

        target_receptacle = min((r for r in spatial_knowledge.locations.values() if name_equal(r.object_type, "CounterTop")), key=spatial_knowledge.get_distance)
        if target_receptacle is None:
            return False, "There is no countertop available"

        setattr(self, "target_object", target_object)
        setattr(self, "target_receptacle", target_receptacle.object_name)
        return True, ""

    def apply(self) -> str:
        return f"motor action: put {self.target_object} on {self.target_receptacle}"
