# flake8: noqa
from production import *

production_name = "PlaceObjectInReceptacle2"


class PlaceObjectInReceptacle2(Production):
    english_description = "IF the current task is to pick up and place a/an <object> in/on a/an <receptacle> AND the <object> is already in the <receptacle> AND the robot's gripper is empty THEN choose special action: 'done'."

    target_task: str = "pick up and place a/an <object> in/on a/an <receptacle>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        # Parse the current task to get the target object and receptacle
        match = re.match(r"pick up and place a/an (.*) in/on a/an (.*)", current_task)
        if not match:
            return False, "The current task does not match the target task format"
        self.target_object, self.target_receptacle = match.groups()

        # Check if the target object is already in the target receptacle
        if not name_equal(object_knowledge[self.target_object].location, self.target_receptacle):
            return False, "The target object is not in the target receptacle"

        # Check if the robot's gripper is empty
        if spatial_knowledge["RobotGripper"].hosting != []:
            return False, "The robot's gripper is not empty"

        return True, ""

    def apply(self) -> str:
        return "special action: done"
