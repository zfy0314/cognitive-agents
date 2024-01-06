# flake8: noqa
from production import *

production_name = "MoveToTargetReceptacle"


class MoveToTargetReceptacle(Production):
    english_description = "IF the current task is to pick up and place a/an <object> in/on a/an <receptacle> AND the robot's gripper has the <object> AND the robot is not at the <receptacle> THEN choose motor action: move to <receptacle>."

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
        self.target_object, self.target_receptacle = match.groups()

        # Check if the robot's gripper has the target object
        gripper = spatial_knowledge["RobotGripper"]
        if not gripper.hosting or not name_equal(gripper.hosting[0].object_type, self.target_object):
            return False, "The robot's gripper does not have the target object"

        # Check if the robot is not at the target receptacle
        if name_equal(current_location, self.target_receptacle):
            return False, "The robot is already at the target receptacle"

        return True, ""

    def apply(self) -> str:
        return f"motor action: move to {self.target_receptacle}"
