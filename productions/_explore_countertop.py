# flake8: noqa
from production import *

production_name = "ExploreCountertop"


class ExploreCountertop(Production):
    english_description = "IF the current task is to put things on the countertops away AND the robot's gripper is empty AND there are unexplored countertops THEN choose 'attend to subtask: go to explore a/an <unexplored_countertop>'."

    target_task: str = "put things on the countertops away"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        # Check if the current task is to put things on the countertops away
        if current_task != self.target_task:
            return False, "The current task is not to put things on the countertops away"

        # Check if the robot's gripper is empty
        if spatial_knowledge["RobotGripper"].hosting != []:
            return False, "The robot's gripper is not empty"

        # Find the first unexplored countertop
        for receptacle in spatial_knowledge.locations.values():
            if name_equal(receptacle.object_type, "CounterTop") and not receptacle.visited:
                self.unexplored_countertop = receptacle.object_name
                return True, ""

        return False, "There are no unexplored countertops"

    def apply(self) -> str:
        return f"attend to subtask: go to explore a/an {self.unexplored_countertop}"
