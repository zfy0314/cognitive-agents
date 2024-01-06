# flake8: noqa
from production import *

production_name = "ExploreReceptacle2"


class ExploreReceptacle2(Production):
    english_description = "IF the current task is to explore a/an <receptacle> AND the robot is in front of the <receptacle> AND the <receptacle> has been fully explored THEN choose special action: 'done'."

    target_task: str = "go to explore a/an <receptacle>"

    def precondition(
        self, 
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge, 
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        # Parse the target receptacle from the current task
        match = re.match(r"go to explore a/an (\w+)", current_task)
        if not match:
            return False, "The current task is not to explore a receptacle"
        self.target_receptacle = match.group(1)

        # Check if the robot is in front of the target receptacle
        if not name_equal(current_location, self.target_receptacle):
            return False, "The robot is not in front of the target receptacle"

        # Check if the target receptacle has been fully explored
        if not spatial_knowledge[self.target_receptacle].visited:
            return False, "The target receptacle has not been fully explored"

        return True, ""

    def apply(self) -> str:
        return "special action: done"
