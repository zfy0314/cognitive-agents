import json
import logging
import os
import pdb  # noqa
import random
import re
from datetime import datetime
from typing import List, NamedTuple, Optional

import fire

import agent
import utils
from domain import Env, name_equal


class TaskBase(NamedTuple):
    task: str
    achievable: bool
    seed: int = 0
    log_dir: str = "logs/"
    floorplan: str = "FloorPlan6"

    def __repr__(self) -> str:

        if hasattr(self, "res"):
            return "[{} in {} starting from {}: {} ({})]\n".format(
                self.task,
                self.floorplan,
                self.starting_location,
                self.res,
                self.message,
            )
        else:
            return "[{} in {}]".format(self.task, self.floorplan)

    def run(self, agent_type: str, max_steps=50, **kwargs) -> bool:

        random.seed(self.seed)
        self.env = Env(floorplan=self.floorplan, **kwargs)
        self._env_preprocess()
        self.starting_location = self.env.current_location
        self.alice = getattr(agent, agent_type)(
            "Alice", env=self.env, parent_log_dir=self.log_dir
        )
        self.res = False
        self.message = "agent run out of max steps"
        try:
            self.alice.run(self.task, max_steps, test=True)
        except utils.Done:
            self.message = "agent chosen done"
            self.res = self.satisfied
        except utils.Quit:
            self.message = "agent chosen quit"
            self.res = not self.achievable
        except Exception as e:
            self.message = "agent encountered {}:{}".format(type(e), e)
            self.res = False
        return self.res

    def _env_preprocess(self):
        self.env.sample_location()

    @property
    def satisfied(self) -> bool:
        raise NotImplementedError


class TaskFind(TaskBase):
    @property
    def satisfied(self) -> bool:
        target_object = self.task.split(" ")[-1].lower()
        for obj_info in self.env.objects_visible_:
            if obj_info["objectType"].lower() == target_object:
                logging.info("Task success! {} is found".format(target_object))
                return True
        return False


class TaskSlice(TaskBase):
    @property
    def satisfied(self) -> bool:
        target_object = self.task.split(" ")[-1].lower()
        for obj_info in self.env.objects_visible_:
            if obj_info["objectType"].lower() == target_object + "sliced":
                logging.info("Task success! {} is sliced".format(target_object))
                return True
        return False


class TaskClear(TaskBase):
    @property
    def satisfied(self) -> bool:
        count = 0
        for obj_info in self.env.objects_:
            if obj_info["pickupable"] and not self.env._get_location(
                obj_info
            ).startswith("CounterTop"):
                count += 1
        return self.total - count > 2 or count == 0

    def _env_preprocess(self):
        self.env.sample_location()
        self.env._sample_objects_on_countertop(5)

        count = 0
        for obj_info in self.env.objects_:
            if obj_info["pickupable"] and not self.env._get_location(
                obj_info
            ).startswith("CounterTop"):
                count += 1
        self.total = count


def test_agent(agent_type: str):

    utils.set_logging("INFO")
    log_dir = "logs/{}".format(agent_type)
    tests = [
        TaskFind("find a/an potato", True, 0, log_dir),
        TaskFind("find a/an mug", True, 1, log_dir),
        TaskFind("find a/an pot", True, 2, log_dir),
        TaskFind("find a/an egg", True, 3, log_dir),
        TaskFind("find a/an garlic", False, 4, log_dir),
        TaskSlice("slice a/an apple", True, 0, log_dir),
        TaskSlice("slice a/an potato", True, 1, log_dir),
        TaskSlice("slice a/an lettuce", True, 2, log_dir),
        TaskSlice("slice a/an tomato", True, 3, log_dir),
        TaskSlice("slice a/an bread", True, 4, log_dir),
        TaskClear("put things on the countertops away", True, 0, log_dir),
        TaskClear("put things on the countertops away", True, 1, log_dir),
        TaskClear("put things on the countertops away", True, 2, log_dir),
    ]

    for test in tests:
        if isinstance(test, TaskClear):
            test.run(agent_type, object_pose_file=None)
        else:
            for object_pose in [
                None,
                "locations/floorplan6_obj1.json",
                "locations/floorplan6_obj2.json",
            ]:
                test.run(agent_type, object_pose_file=object_pose)


class Trainer:
    receptacles = [
        "cabinet",
        "drawer",
        "fridge",
        "sinkbasin",
        "stoveburner",
    ]
    objects = [
        "lettuce",
        "potato",
        "apple",
        "bread",
        "cup",
        "mug",
        "pot",
        "plate",
        "tomato",
        "knife",
        "spoon",
        "fork",
        "egg",
        "kettle",
        "pan",
        "bowl",
        "spatula",
    ]
    sliceables = [
        "bread",
        "apple",
        "tomato",
        "lettuce",
        "potato",
    ]

    def __init__(
        self, log_dir: Optional[str] = None, production_dir: Optional[str] = None
    ):

        timestamp = datetime.now().strftime("%m%d%y%H%M%S")
        if log_dir is None:
            self.log_dir = os.path.join("logs", "training_" + timestamp)
        else:
            self.log_dir = log_dir
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

        if production_dir is None:
            self.production_dir = "productions_" + timestamp
        else:
            self.production_dir = production_dir
        if not os.path.isdir(self.production_dir):
            os.makedirs(self.production_dir)
        init_file = os.path.join(self.production_dir, "__init__.py")
        if not os.path.isfile(init_file):
            with open(init_file, "w"):
                pass

        self.critic_dir = os.path.join(self.log_dir, "critics")
        if not os.path.isdir(self.critic_dir):
            os.makedirs(self.critic_dir)

        self.task_description_file = os.path.join(
            self.log_dir, "task_descriptions.json"
        )
        if os.path.isfile(self.task_description_file):
            with open(self.task_description_file, "r") as fin:
                self.task_descriptions = json.load(fin)
        else:
            self.task_descriptions = {}
        with open("prompts/critic/system.txt", "r") as fin:
            self.critic_system = fin.read()
        with open("prompts/critic/user_summary.txt", "r") as fin:
            self.critic_user = fin.read()

        self.env = Env(
            floorplan="FloorPlan5", object_pose_file="locations/floorplan5_obj1.json"
        )

    def _init_env(self, task: str):

        self.env.reset()
        self.env.sample_location()
        if task == "put things on the countertops away":
            self.env._sample_objects_on_countertop(5)
        else:
            match = re.fullmatch(r"pick up and place a/an (.*) in/on a/an (.*)", task)
            if match is not None:
                target_obj = match.group(1)
                target_recep = match.group(2)
                for obj_info in self.env.objects_:
                    if name_equal(
                        self.env._get_location(obj_info), target_recep
                    ) and not name_equal(obj_info["name"], target_obj):
                        self.env.api_step(
                            "DisableObject", objectId=obj_info["objectId"]
                        )

    def train_task(self, task: str, num: int = 5):

        random.seed(0)

        logging.info("starting training for {}".format(task))
        for i in range(num):
            if "<receptacle>" in task:
                recep_name = random.choice(self.receptacles)
            else:
                recep_name = ""
            if "<object>" in task:
                if recep_name == "stoveburner" and task.startswith("pick up and place"):
                    # simulator constraints, it does not work with other objects
                    obj_name = random.choice(["pot", "pan"])
                else:
                    obj_name = random.choice(self.objects)
                    while recep_name == obj_name:
                        obj_name = random.choice(self.objects)
            else:
                obj_name = ""
            if "<sliceable>" in task:
                slice_name = random.choice(self.sliceables)
                while obj_name == slice_name or recep_name == slice_name:
                    slice_name = random.choice(self.sliceables)
            else:
                slice_name = ""
            specific_task = (
                task.replace("<receptacle>", recep_name)
                .replace("<object>", obj_name)
                .replace("<sliceable>", slice_name)
            )
            logging.info("training on {}".format(specific_task))

            self._init_env(specific_task)
            self.env._update_screen(self.env.event)
            # pdb.set_trace()

            alice = agent.Agent(
                name="Alice",
                env=self.env,
                production_dir=self.production_dir,
                parent_log_dir=self.log_dir,
                log_dir="{}_{:02d}".format(task.split(" ")[0], i),
                task_description_file=self.task_description_file,
            )
            alice.run(utils.Task(specific_task, task), max_steps=200)
            alice.save_state_diagram()
            alice.save_weights(
                os.path.join(self.production_dir, "production_weights.json")
            )
            agent.Monitor.reset_state_diagram()
        alice.save_weights(
            os.path.join(
                self.production_dir,
                "production_weights_{}.json".format(
                    datetime.now().strftime("%m%d%y%H%M%S")
                ),
            )
        )
        messages = [
            {"role": "system", "content": self.critic_system},
            {
                "role": "user",
                "content": self.critic_user.format(
                    task=task,
                    productions="\n".join(
                        " * " + x for x in alice._get_production_descriptions(task)
                    ),
                    subtasks="\n".join("   - " + x for x in alice._get_subtasks()),
                ),
            },
        ]
        response = utils.GPTquery(
            messages,
            os.path.join(
                self.critic_dir,
                "".join(
                    x.lower() if x.isalnum() else "_" if x == " " else "" for x in task
                ),
            ),
            gpt4=True,
        )
        try:
            condition = re.search(
                r"\[End Condition]\nThe task ends when ([\s\S]*?)\n",
                response,
            ).group(1)
            self.task_descriptions[task] = condition
            with open(self.task_description_file, "w") as fout:
                json.dump(self.task_descriptions, fout, indent=2)
            with open(
                self.task_description_file.replace(
                    ".json", "_{}.json".format(task.split(" ")[0])
                ),
                "w",
            ) as fout:
                json.dump(self.task_descriptions, fout, indent=2)
        except AttributeError:
            logging.warning("summary cannot be parsed, setting to None")
        pdb.set_trace()

    def train_curriculum(self, curriculum: List[str], num: int = 10):

        for task in curriculum:
            self.train_task(task, num)


def train_agent():

    utils.set_logging("INFO")
    T = Trainer("logs/training_log", "productions")
    T.train_curriculum(
        [
            "go to explore a/an <receptacle>",
            "find a/an <object>",
            "pick up and place a/an <object> in/on a/an <receptacle>",
            "slice a/an <sliceable>",
            "put things on the countertops away",
        ],
        10,
    )


if __name__ == "__main__":
    fire.Fire()
