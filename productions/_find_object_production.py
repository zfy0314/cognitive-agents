# flake8: noqa
from production import *

production_name = "FindObjectProduction"


class FindObjectProduction(Production):
    english_description = "IF the current task is to find a/an <object> AND the robot's gripper is empty AND there are unexplored receptacles in the kitchen THEN choose 'attend to subtask: go to explore a/an <closest_unexplored_receptacle>'."

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
        match = re.match(r"find a/an (\w+)", current_task)
        if not match:
            return False, "The current task is not to find an object."
        target_object = match.group(1)

        # Check if the robot's gripper is empty
        if spatial_knowledge["RobotGripper"].hosting != []:
            return False, "The robot's gripper is not empty."

        # Find the closest unexplored receptacle
        closest_unexplored_receptacle = None
        min_distance = float("inf")
        for receptacle in spatial_knowledge.locations.values():
            if not receptacle.visited and spatial_knowledge.get_distance(receptacle.object_name) < min_distance:
                closest_unexplored_receptacle = receptacle
                min_distance = spatial_knowledge.get_distance(receptacle.object_name)
        if closest_unexplored_receptacle is None:
            return False, "There are no unexplored receptacles in the kitchen."

        # Set variable bindings
        setattr(self, "target_object", target_object)
        setattr(self, "closest_unexplored_receptacle", closest_unexplored_receptacle)

        return True, ""

    def apply(self) -> str:
        return f"attend to subtask: go to explore a/an {self.closest_unexplored_receptacle.object_name}"
