# flake8: noqa
from production import *

production_name = "FindObjectProduction2"


class FindObjectProduction2(Production):
    english_description = "IF the current task is to pick up and place a/an <object> in/on a/an <receptacle> AND the robot's gripper is empty AND the <object> has not been located THEN choose 'attend to subtask: find a/an <object>'."

    target_task: str = "pick up and place a/an <object> in/on a/an <receptacle>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        # Parse the current task to get the target object and target receptacle
        match = re.match(r"pick up and place a/an (.*) in/on a/an (.*)", current_task)
        if not match:
            return (False, "The current task does not match the target task")
        self.target_object, self.target_receptacle = match.groups()

        # Check if the robot's gripper is empty
        if spatial_knowledge["RobotGripper"].hosting != []:
            return (False, "The robot's gripper is not empty")

        # Check if the target object has not been located
        if self.target_object in object_knowledge:
            return (False, f"The {self.target_object} has been located")

        return (True, "")

    def apply(self) -> str:
        return f"attend to subtask: find a/an {self.target_object}"
