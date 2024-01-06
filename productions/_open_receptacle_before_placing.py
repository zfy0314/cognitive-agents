# flake8: noqa
from production import *

production_name = "OpenReceptacleBeforePlacing"


class OpenReceptacleBeforePlacing(Production):
    english_description = "IF the current task is to pick up and place a/an <object> in/on a/an <receptacle> AND the robot's gripper has the <object> AND the robot is at the <receptacle> AND the <receptacle> is closed THEN choose motor action: open <receptacle>."

    target_task: str = "pick up and place a/an <object> in/on a/an <receptacle>"

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
        # Parse the target object and target receptacle from the current task
        match = re.match(r"pick up and place a/an (.*) in/on a/an (.*)", current_task)
        if not match:
            return (False, "The current task does not match the target task")
        target_object, target_receptacle = match.groups()

        # Set the target_receptacle as an attribute of the object
        setattr(self, 'target_receptacle', target_receptacle)

        # Check if the robot's gripper has the target object
        if target_object not in spatial_knowledge["RobotGripper"]:
            return (False, "The robot's gripper does not have the target object")

        # Check if the robot is at the target receptacle
        if not name_equal(current_location, target_receptacle):
            return (False, "The robot is not at the target receptacle")

        # Check if the target receptacle is closed
        if object_knowledge[target_receptacle].opened:
            return (False, "The target receptacle is not closed")

        return (True, "")

    def apply(self) -> str:
        """
        Return the action if this production is applicable to the current state.
        This corresponds to the THEN part of the production rule.
        """
        return "motor action: open " + self.target_receptacle
