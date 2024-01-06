# flake8: noqa
from production import *

production_name = "ClearReceptacleDone"


class ClearReceptacleDone(Production):
    english_description = "IF the current task is to clear objects from a/an <receptacle_type> AND all <receptacle_type> have been explored and are empty AND the robot's gripper is empty THEN choose special action: 'done'."

    target_task: str = "put things on the countertops away"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:

        # Parse the receptacle type from the current task
        match = re.match(r"put things on the (.+) away", current_task)
        if not match:
            return False, "The current task is not to clear objects from a receptacle"
        self.receptacle_type = match.group(1)

        # Find all receptacles of the specified type
        self.all_receptacles = [receptacle for receptacle in spatial_knowledge.locations.values() if name_equal(receptacle.object_type, self.receptacle_type)]

        # Check if all receptacles of the specified type have been explored and are empty
        for receptacle in self.all_receptacles:
            if not receptacle.visited or receptacle.get_hosting():
                return False, f"{receptacle.object_name} has not been explored or is not empty"

        # Check if the robot's gripper is empty
        if spatial_knowledge["RobotGripper"].hosting:
            return False, "The robot's gripper is not empty"

        return True, ""

    def apply(self) -> str:
        return "special action: done"
