# flake8: noqa
from production import *

production_name = "FindObject"


class FindObject(Production):
    english_description = "IF the current task is to find a/an <object> AND the robot's gripper is empty AND there is an unexplored <receptacle> that is commonly associated with the <object> THEN choose motor action: move to <receptacle>."

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
        match = re.match(r"find a/an (.+)", current_task)
        if not match:
            return False, "The current task is not to find an object"
        target_object = match.group(1)

        # Check if the robot's gripper is empty
        if spatial_knowledge["RobotGripper"].hosting != []:
            return False, "The robot's gripper is not empty"

        # Find an unexplored receptacle that is commonly associated with the target object
        for receptacle in spatial_knowledge.locations.values():
            if OracleQuery(f"{target_object} are commonly stored in {receptacle.object_type}") and not receptacle.visited:
                self.target_receptacle = receptacle
                return True, ""

        return False, "No unexplored receptacle is commonly associated with the target object"

    def apply(self) -> str:
        return f"motor action: move to a/an {self.target_receptacle.object_name}"
