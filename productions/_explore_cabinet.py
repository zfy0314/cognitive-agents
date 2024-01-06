# flake8: noqa
from production import *

production_name = "ExploreCabinet"


class ExploreCabinet(Production):
    english_description = "IF the current task is to put things on the countertops away AND the robot's gripper is empty AND the robot has already explored all the countertops AND there are unexplored cabinets THEN choose attend to subtask: go to explore a/an <unexplored_cabinet>."

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

        # Check if the robot has already explored all the countertops
        for receptacle in spatial_knowledge.locations.values():
            if name_equal(receptacle.object_type, "CounterTop") and not receptacle.visited:
                return False, "The robot has not explored all the countertops"

        # Find an unexplored cabinet
        for receptacle in spatial_knowledge.locations.values():
            if name_equal(receptacle.object_type, "Cabinet") and not receptacle.visited:
                self.unexplored_cabinet = receptacle.object_name
                return True, ""

        return False, "There are no unexplored cabinets"

    def apply(self) -> str:
        # Choose to attend to the subtask of exploring the unexplored cabinet
        return f"attend to subtask: go to explore a/an {self.unexplored_cabinet}"
