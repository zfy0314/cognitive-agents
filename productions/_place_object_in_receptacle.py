# flake8: noqa
from production import *

production_name = "PlaceObjectInReceptacle"


class PlaceObjectInReceptacle(Production):
    english_description = "IF the current task is to pick up and place a/an <object> in/on a/an <receptacle> AND the robot is holding the <object> AND the robot is in front of the <receptacle> AND the <receptacle> is empty THEN choose motor action: put <object> on <receptacle>."

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
        if not match:
            return False, "The current task does not match the target task format"
        self.target_object, self.target_receptacle = match.groups()

        # Check if the robot is holding the target object
        if not spatial_knowledge["RobotGripper"].hosting or not name_equal(spatial_knowledge["RobotGripper"].hosting[0].object_name, self.target_object):
            return False, "The robot is not holding the target object"

        # Check if the robot is in front of the target receptacle
        if not name_equal(current_location, self.target_receptacle):
            return False, "The robot is not in front of the target receptacle"

        # Check if the target receptacle is empty
        if spatial_knowledge[self.target_receptacle].hosting != []:
            return False, "The target receptacle is not empty"

        return True, ""

    def apply(self) -> str:
        return "motor action: put " + self.target_object + " on " + self.target_receptacle
