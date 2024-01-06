# flake8: noqa
from production import *

production_name = "FindObjectInGripper"


class FindObjectInGripper(Production):
    english_description = "IF the current task is to find a/an <object> AND the robot's gripper has <object> THEN choose special action: 'done'."

    target_task: str = "find a/an <object>"

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
        # Parse the current task to get the target object
        match = re.match(r"find a/an (.+)", current_task)
        if match is None:
            return (False, "The current task is not to find an object.")
        self.target_object = match.group(1)

        # Check if the robot's gripper has the target object
        for obj in spatial_knowledge["RobotGripper"].hosting:
            if name_equal(obj.object_type, self.target_object):
                return (True, "")
        
        return (False, "The robot's gripper does not have the target object.")

    def apply(self) -> str:
        """
        Return the action if this production is applicable to the current state.
        This corresponds to the THEN part of the production rule.
        """
        return "special action: done"
