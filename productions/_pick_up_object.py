# flake8: noqa
from production import *

production_name = "PickUpObject"


class PickUpObject(Production):
    english_description = "IF the current task is to pick up and place a/an <object> in/on a/an <receptacle> AND the robot is located in front of a countertop that has the <object> on it AND the robot's gripper is empty THEN choose motor action: pick up <object>."

    target_task: str = "pick up and place a/an <object> in/on a/an <receptacle>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:

        # Parse the target object and receptacle from the current task
        match = re.match(r"pick up and place a/an (.*) in/on a/an (.*)", current_task)
        if match is None:
            return False, "The current task does not match the target task"
        target_object, target_receptacle = match.groups()

        # Check if the robot is located in front of a countertop that has the target object on it
        if not name_equal(current_location, "CounterTop") or target_object not in spatial_knowledge[current_location]:
            return False, "The robot is not located in front of a countertop that has the target object on it"

        # Check if the robot's gripper is empty
        if spatial_knowledge["RobotGripper"].hosting != []:
            return False, "The robot's gripper is not empty"

        # Set variable bindings
        setattr(self, "target_object", target_object)

        return True, ""

    def apply(self) -> str:
        return "motor action: pick up " + self.target_object
