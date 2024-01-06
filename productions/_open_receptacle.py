# flake8: noqa
from production import *

production_name = "OpenReceptacle"


class OpenReceptacle(Production):
    english_description = "IF the current task is to find a/an <object> AND the robot is in front of a closed <receptacle> AND the robot's gripper is empty THEN choose motor action: open <receptacle>."

    target_task: str = "find a/an <object>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:

        # Parse the target object from the current task
        match = re.match(r"find a/an (\w+)", current_task)
        if not match:
            return (False, "The current task is not to find an object.")
        self.target_object = match.group(1)

        # Check if the robot is in front of a closed receptacle
        if not object_knowledge[spatial_knowledge[current_location]].opened == False:
            return (False, "The robot is not in front of a closed receptacle.")
        self.target_receptacle = current_location

        # Check if the robot's gripper is empty
        if not spatial_knowledge["RobotGripper"].hosting == []:
            return (False, "The robot's gripper is not empty.")

        return (True, "")

    def apply(self) -> str:
        return f"motor action: open {self.target_receptacle}"
