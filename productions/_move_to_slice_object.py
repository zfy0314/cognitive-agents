# flake8: noqa
from production import *

production_name = "MoveToSliceObject"


class MoveToSliceObject(Production):
    english_description = "IF the current task is to slice a/an <object> AND the robot is holding a knife AND the <object> is not at the robot's current location THEN choose motor action: move to <object_location>."

    target_task: str = "slice a/an <sliceable>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        """
        Test if the precondition is met given the current information.
        Set variable bindings if necessary.
        This corresponds to the IF part of the production rule.
        Return (True, "") if the precondition is met.
        Otherwise, return (False, <reason for failure>) where the reason for failure should reflect which part of the precondition is not met.
        """
        # Check if the current task is to slice a/an <object>
        if not current_task.startswith("slice a/an "):
            return (False, "The current task is not to slice an object.")
        
        # Parse the target object from the current task
        target_object = re.search(r'slice a/an (.+)', current_task).group(1)
        
        # Check if the robot is holding a knife
        if spatial_knowledge["RobotGripper"].hosting is None or spatial_knowledge["RobotGripper"].hosting[0].object_type != "Knife":
            return (False, "The robot is not holding a knife.")
        
        # Check if the target object is known
        if target_object not in object_knowledge:
            return (False, "The target object is not known.")
        
        # Get the location of the target object
        target_location = object_knowledge[target_object].location
        
        # Check if the target object is at the robot's current location
        if target_location == current_location:
            return (False, "The target object is at the robot's current location.")
        
        # Set the target location as a variable
        setattr(self, "target_location", target_location)
        
        return (True, "")

    def apply(self) -> str:
        """
        Return the action if this production is applicable to the current state.
        This corresponds to the THEN part of the production rule.
        """
        # Move to the target location
        return f"motor action: move to {self.target_location}"
