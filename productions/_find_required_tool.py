# flake8: noqa
from production import *

production_name = "FindRequiredTool"


class FindRequiredTool(Production):
    english_description = "IF the current task is to slice a/an <sliceable> AND the <sliceable> is located on a countertop AND the robot's gripper is empty AND there is no <required_tool> in the robot's spatial knowledge THEN choose 'attend to subtask: find a/an <required_tool>'."

    target_task: str = "slice a/an <sliceable>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        # Parse the sliceable and required tool from the current task
        if not current_task.startswith("slice a/an "):
            return False, "The current task is not to slice an object."
        sliceable, required_tool = current_task.split(" ")[-1], "knife"

        # Check if the sliceable object is located on a countertop
        sliceable_location = object_knowledge[sliceable].location
        if not name_equal(sliceable_location, "CounterTop"):
            return False, f"The {sliceable} is not located on a countertop."

        # Check if the robot's gripper is empty
        gripper = spatial_knowledge["RobotGripper"]
        if gripper.hosting != []:
            return False, "The robot's gripper is not empty."

        # Check if the required tool is not in the robot's spatial knowledge
        if required_tool in spatial_knowledge:
            return False, f"The {required_tool} is already in the robot's spatial knowledge."

        return True, ""

    def apply(self) -> str:
        return "attend to subtask: find a/an knife"
