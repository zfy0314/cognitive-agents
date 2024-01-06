# flake8: noqa
from production import *

production_name = "ExploreReceptacle"


class ExploreReceptacle(Production):
    english_description = "IF the current task is to explore a/an <receptacle> AND the <receptacle> is not at the current location AND the robot's gripper is empty THEN choose 'motor action: move to <receptacle>'."

    target_task: str = "go to explore a/an <receptacle>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        # Parse the target receptacle from the current task
        match = re.match(r"go to explore a/an (\w+)", current_task)
        if not match:
            return False, "The current task is not to explore a receptacle."
        self.target_receptacle = match.group(1)

        # Get the current receptacle based on the current location
        current_receptacle = spatial_knowledge[current_location].object_type

        # Check if the target receptacle is not at the current location
        if name_equal(current_receptacle, self.target_receptacle):
            return False, "The target receptacle is at the current location."

        # Check if the robot's gripper is empty
        if spatial_knowledge["RobotGripper"].hosting != []:
            return False, "The robot's gripper is not empty."

        # All preconditions are met
        return True, ""

    def apply(self) -> str:
        # The action is to move to the target receptacle
        return "motor action: move to " + self.target_receptacle
