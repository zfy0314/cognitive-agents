# flake8: noqa
from production import *

production_name = "PutAwayObjects"


class PutAwayObjects(Production):
    english_description = "IF the current task is to put things on the countertops away AND the robot's gripper is empty AND there are objects on the countertops AND there are empty cabinets THEN choose 'attend to subtask: pick up and place a/an <object_on_countertop> in/on a/an <empty_cabinet>'."

    target_task: str = "put things on the countertops away"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:

        if current_task != self.target_task:
            return False, "The current task is not to put things on the countertops away"

        if spatial_knowledge["RobotGripper"].hosting != []:
            return False, "The robot's gripper is not empty"

        for obj in object_knowledge.objects.values():
            if obj.location in spatial_knowledge.locations and name_equal(spatial_knowledge.locations[obj.location].object_type, "CounterTop"):
                self.object_on_countertop = obj
                break
        else:
            return False, "There are no objects on the countertops"

        for receptacle in spatial_knowledge.locations.values():
            if name_equal(receptacle.object_type, "Cabinet") and receptacle.hosting == []:
                self.empty_cabinet = receptacle
                break
        else:
            return False, "There are no empty cabinets"

        return True, ""

    def apply(self) -> str:
        return f"attend to subtask: pick up and place a/an {self.object_on_countertop.object_name} in/on a/an {self.empty_cabinet.object_name}"
