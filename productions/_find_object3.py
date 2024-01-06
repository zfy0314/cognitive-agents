# flake8: noqa
from production import *

production_name = "FindObject3"


class FindObject3(Production):
    english_description = "IF the current task is to find a/an <object> AND the <object> is located in a/an <receptacle> AND the robot is in front of the <receptacle> AND the robot's gripper is empty THEN choose motor action: pick up <object>."

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
        target_object = re.search(r'find a/an (\w+)', current_task)
        if target_object is None:
            return (False, "The current task is not to find an object.")
        target_object = target_object.group(1)

        # Check if the target object is in the object knowledge
        if target_object not in object_knowledge:
            return (False, "The target object is not in the object knowledge.")
        
        # Get the target receptacle that contains the target object
        target_receptacle = object_knowledge[target_object].location

        # Check if the robot is in front of the target receptacle
        if not name_equal(current_location, target_receptacle):
            return (False, "The robot is not in front of the target receptacle.")

        # Check if the robot's gripper is empty
        if spatial_knowledge["RobotGripper"].hosting != []:
            return (False, "The robot's gripper is not empty.")

        # Set the target object and target receptacle as attributes of the class
        setattr(self, "target_object", target_object)
        setattr(self, "target_receptacle", target_receptacle)

        return (True, "")

    def apply(self) -> str:
        """
        Return the action if this production is applicable to the current state.
        """
        return f"motor action: pick up {self.target_object}"
