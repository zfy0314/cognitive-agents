# flake8: noqa
from production import *

production_name = "ExploreReceptacle4"


class ExploreReceptacle4(Production):
    english_description = "IF the current task is to explore a/an <receptacle> AND the robot is not at the location of <receptacle> AND the robot's gripper is empty THEN choose motor action: move to <receptacle>."

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
        if match:
            self.target_receptacle = match.group(1)
            if not name_equal(current_location, self.target_receptacle) and spatial_knowledge["RobotGripper"].hosting == []:
                return (True, "")
            else:
                return (False, "Robot is either at the location of the target receptacle or its gripper is not empty.")
        else:
            return (False, "Current task is not to explore a receptacle.")

    def apply(self) -> str:
        return "motor action: move to " + self.target_receptacle
