import json
import os
from typing import Dict, Optional, Tuple

import utils
from domain import name_equal  # noqa
from domain import NameCompatibleList, ObjectKnowledge, SpatialKnowledge

re = utils.re
_gamma = 0.95


list_of_large_receptacles = NameCompatibleList(
    [
        "CounterTop",
        "Cabinet",
        "Drawer",
        "Fridge",
        "SinkBasin",
    ]
)


class Production:
    english_description: str
    target_task: str = ""
    utility: Optional[float] = 0
    num_calls: int = 0

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Production)
            and self.english_description == other.english_description
        )

    def __le__(self, other) -> bool:

        return (
            isinstance(other, Production)
            and other.english_description < self.english_description
        )

    def __hash__(self) -> int:
        return hash(self.english_description)

    def precondition(
        self,
        current_task: str,
        current_location: str,
        previous_tasks: Dict[str, bool],
        spatial_knowledge: SpatialKnowledge,
        object_knowledge: ObjectKnowledge,
    ) -> Tuple[bool, str]:
        raise NotImplementedError

    def apply(self) -> str:
        raise NotImplementedError

    def update_success(self, interval: int):

        if self.utility is not None:
            self.utility = self.utility * self.num_calls + _gamma ** interval
            self.num_calls += 1
            self.utility /= self.num_calls

    def disable(self):

        self.utility = None

    def is_disabled(self) -> bool:

        return self.utility is None


class OracleQueryClass:
    def __init__(self, log_file: str):

        self.log_file = log_file
        self.database = {}
        self.reload()

    def reload(self):

        if os.path.isfile(self.log_file):
            with open(self.log_file, "r") as fin:
                self.database = json.load(fin)

    def __call__(self, statement: str) -> bool:

        if statement not in self.database.keys():
            messages = [
                {
                    "role": "user",
                    "content": "Evaluate the correctness of the following statement: '"
                    + statement
                    + "', reply only True, False, or Unknown",
                }
            ]
            response = utils.GPTquery(messages, gpt4=True)
            if response.lower().strip() == "true":
                self.database[statement] = True
            elif response.lower().strip() == "false":
                self.database[statement] = False
            else:
                raise ValueError("got unknown statement: {}".format(statement))
            with open(self.log_file, "w") as fout:
                json.dump(self.database, fout, indent=2)

        return self.database[statement]


OracleQuery = OracleQueryClass("oracle_database.json")
