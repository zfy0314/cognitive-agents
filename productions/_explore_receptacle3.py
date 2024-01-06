# flake8: noqa
from production import *

production_name = "ExploreReceptacle3"


class ExploreReceptacle3(Production):
    english_description = "IF the current task is to explore a <receptacle> AND the robot is at the location of the <receptacle> AND the <receptacle> is closed and unexplored AND the robot's gripper is empty THEN choose 'motor action: open <receptacle>'."

    target_task: str = "go to explore a/an <receptacle>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        match = re.match(r"go to explore a/an (\w+)", current_task)
        if not match:
            return False, "The current task is not to explore a receptacle"
        self.target_receptacle = match.group(1)
        if not name_equal(current_location, self.target_receptacle):
            return False, "The robot is not at the location of the target receptacle"
        if object_knowledge[self.target_receptacle].opened or spatial_knowledge[self.target_receptacle].visited:
            return False, "The target receptacle is not closed or has been explored"
        if spatial_knowledge["RobotGripper"].hosting != []:
            return False, "The robot's gripper is not empty"
        return True, ""

    def apply(self) -> str:
        return "motor action: open " + self.target_receptacle
