# flake8: noqa
from production import *

production_name = "FindObject2"


class FindObject2(Production):
    english_description = "IF the current task is to find a/an <object> AND the <object> is located in <location> AND the robot is not at the location of the <object> AND the robot's gripper is empty THEN choose motor action: move to <location>."

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
        target_object_type = match.group(1)

        # Get the target object from the object knowledge
        target_object = object_knowledge[target_object_type]
        if not target_object:
            return False, f"The robot does not know about any {target_object_type}"

        # Get the target location from the spatial knowledge
        target_location = spatial_knowledge[target_object.location]
        if not target_location:
            return False, f"The robot does not know about the location {target_object.location}"

        # Check if the robot is at the target location
        if current_location == target_location.object_name:
            return False, "The robot is already at the location of the target object"

        # Check if the robot's gripper is empty
        if spatial_knowledge["RobotGripper"].hosting != []:
            return False, "The robot's gripper is not empty"

        # Set the target object and target location as attributes of the production
        setattr(self, "target_object", target_object)
        setattr(self, "target_location", target_location)

        return True, ""

    def apply(self) -> str:
        return f"motor action: move to {self.target_location.object_name}"
