# flake8: noqa
from production import *

production_name = "MoveToPickUpObject"


class MoveToPickUpObject(Production):
    english_description = "IF the current task is to pick up and place a/an <object> in/on a/an <receptacle> AND the robot's gripper is empty AND the robot is not at the location of the <object> THEN choose motor action: move to <object_location>."

    target_task: str = "pick up and place a/an <object> in/on a/an <receptacle>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        # Parse the target object and target receptacle from the current task
        match = re.match(r"pick up and place a/an (.*) in/on a/an (.*)", current_task)
        if not match:
            return False, "The current task does not match the target task"
        target_object, target_receptacle = match.groups()

        # Check if the robot's gripper is empty
        if spatial_knowledge["RobotGripper"].hosting != []:
            return False, "The robot's gripper is not empty"

        # Check if the robot is at the location of the target object
        target_object_location = object_knowledge[target_object].location
        if current_location == target_object_location:
            return False, "The robot is already at the location of the target object"

        # Set the target object and target object location as attributes of the production
        setattr(self, "target_object", target_object)
        setattr(self, "target_object_location", target_object_location)

        return True, ""

    def apply(self) -> str:
        return f"motor action: move to {self.target_object_location}"
