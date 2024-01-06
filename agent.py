import functools
import importlib
import json
import logging
import math
import os
import pdb
import pickle
import random
import re
import shutil  # noqa
from collections import deque
from datetime import datetime
from glob import glob
from pprint import pformat
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple, Union

from graphviz import Digraph

import domain
import production
import utils

random.seed(0)


class Monitor:
    topography: Dict[str, Set[str]] = {}
    monitor_connections: Dict[Tuple[str, str], str] = {}
    monitor_dict: Dict[str, "Monitor"] = {}
    monitor_name_stack: List[str] = deque()
    leaf_count: int = 0

    # helper constants
    done_state: int = -1
    quit_state: int = -2

    @staticmethod
    def special_states():
        return {str(Monitor.done_state), str(Monitor.quit_state)}

    @staticmethod
    def render(filename: str):

        Monitor.leaf_count = 0
        children_states = functools.reduce(
            set.union, Monitor.topography.values(), set()
        )
        G = Digraph("G", filename=filename)
        for k in Monitor.topography.keys():
            if k not in children_states:
                G.subgraph(Monitor._render(k))
        for (parent, _), sub_monitor_name in Monitor.monitor_connections.items():
            sub_monitor = Monitor.monitor_dict[sub_monitor_name]
            if sub_monitor.root is not None:
                G.edge(parent, str(sub_monitor.root))

        G.render()

    @staticmethod
    def _render(monitor_name: str) -> Digraph:

        monitor = Monitor.monitor_dict[monitor_name]
        G = Digraph(monitor_name)
        G.attr(label=monitor.task)

        for name in Monitor.topography[monitor_name]:
            G.subgraph(Monitor._render(name))

        for state in monitor.state_graph.keys():
            if state not in Monitor.special_states():
                if "\t{}".format(state) not in G.body:
                    G.node(state, label="", shape="circle")
        for state, connections in monitor.state_graph.items():
            for action, neighbor in connections.items():
                if neighbor is None:
                    neighbor = "leaf_{}".format(Monitor.leaf_count)
                    G.node(
                        neighbor,
                        label="",
                        style="filled",
                        shape="square",
                        color="lightgrey",
                    )
                    Monitor.leaf_count += 1
                if (state, action) in Monitor.monitor_connections.keys():
                    sub_monitor_name = Monitor.monitor_connections[(state, action)]
                    sub_monitor = Monitor.monitor_dict[sub_monitor_name]
                    edge_style = "dashed" if sub_monitor.failure else "solid"
                    width = "3" if sub_monitor.success or sub_monitor.failure else "1"
                    G.edge(
                        sub_monitor.current_state,
                        neighbor,
                        style=edge_style,
                        penwidth=width,
                    )
                elif neighbor not in Monitor.special_states():
                    G.edge(state, neighbor, action)
        return G

    @staticmethod
    def register(monitor: "Monitor") -> str:

        name = "cluster_{}".format(len(Monitor.topography))
        if len(Monitor.monitor_name_stack) > 0:
            current_name = Monitor.monitor_name_stack[-1]
            Monitor.topography[current_name].add(name)
            current_monitor = Monitor.monitor_dict[current_name]
            Monitor.monitor_connections[
                (current_monitor.current_state, current_monitor.current_action)
            ] = name
        Monitor.topography[name] = set()
        Monitor.monitor_dict[name] = monitor
        Monitor.monitor_name_stack.append(name)
        return name

    @staticmethod
    def end(monitor_name: str):
        Monitor.monitor_name_stack.pop()

    @staticmethod
    def reset_state_diagram():
        Monitor.topography: Dict[str, Set[str]] = {}
        Monitor.monitor_connections: Dict[Tuple[str, str], str] = {}
        Monitor.monitor_dict: Dict[str, "Monitor"] = {}
        Monitor.monitor_name_stack: List[str] = deque()
        Monitor.leaf_count: int = 0

    def __init__(self, start_state: Optional[int] = None, task: str = ""):

        self.graph: Dict[
            int, Dict[Tuple[str, production.Production], Optional[int]]
        ] = {}
        self.last_production: Tuple[int, [str, production.Production]] = None
        self.root = start_state
        self.task = task
        self.name = Monitor.register(self)

    def __repr__(self) -> str:

        res = ""
        for key, connections in self.graph.items():
            res += "({})\n".format(key)
            for (a, p), v in connections.items():
                res += " -> [{}] ({})\n".format(a, v)
            res += "\n"
        return res

    @property
    def state_graph(self) -> Dict[str, Dict[str, Optional[str]]]:

        return {
            str(s): {a: None if s2 is None else str(s2) for (a, p), s2 in k.items()}
            for s, k in self.graph.items()
        }

    @property
    def current_state(self) -> str:

        return "" if self.last_production is None else str(self.last_production[0])

    @property
    def current_action(self) -> str:
        return "" if self.last_production is None else self.last_production[1][0]

    @property
    def success(self) -> bool:
        return Monitor.done_state in self.graph.keys()

    @property
    def failure(self) -> bool:
        return Monitor.quit_state in self.graph.keys()

    def close(self):
        Monitor.end(self.name)

    def update(
        self,
        state: int,
        available_productions: List[Tuple[str, production.Production]],
        chosen_production: Optional[Tuple[str, production.Production]],
    ) -> List[str]:
        """Return looping actions"""

        if self.root is None:
            self.root = state

        if self.last_production is not None and self.last_production[-1] is not None:
            last_state, last_chosen = self.last_production
            self.graph[last_state][last_chosen] = state

        if state not in self.graph.keys():
            self.graph[state] = {}
        for key in available_productions:
            if key not in self.graph[state]:
                self.graph[state][key] = None
        if str(state) not in self.special_states() and chosen_production is not None:
            self.last_production = (state, chosen_production)

        if self.is_looped(state):
            logging.debug("looped state detected")
            return [x[0] for x in self.graph[state].keys()]
        else:
            return []

    def is_looped(self, state: int, visited: Set[int] = set()) -> bool:

        if all([x in visited for x in self.graph[state].values()]):
            return True
        if None in self.graph[state].values():
            return False
        for successor in self.graph[state].values():
            if successor not in visited and not self.is_looped(
                successor, visited.union({state})
            ):
                return False
        return True

    def find_path(
        self, goal_state: int, start_state: Optional[int] = None
    ) -> List[Tuple[int, Tuple[str, production.Production]]]:

        if start_state is None:
            start_state = self.root

        todo = deque([(start_state, [])])
        visited = set()
        while len(todo) != 0:
            state, path = todo.popleft()
            if state == goal_state:
                return path
            if state not in visited:
                visited.add(state)
                for key, value in self.graph[state].items():
                    if value is not None:
                        todo.append((value, path + [(state, key)]))

        logging.warning(
            "no path found from {} to {}, returning []".format(start_state, goal_state)
        )
        logging.debug("connectivity diagram")
        logging.debug(str(self))
        return []


class TaskStack:

    stack: List[Tuple[utils.Task, Monitor]]

    def __init__(self, task: Optional[str] = None):

        self.stack = deque()
        self.state_hash = None
        self.last_production = None
        if task is not None:
            self.append(task)

    def __len__(self):
        return len(self.stack)

    def __repr__(self) -> str:
        return str([x[0] for x in self.stack])

    @property
    def current_task(self) -> utils.Task:
        return self.stack[-1][0]

    def append(self, task: Union[utils.Task, str]):

        if isinstance(task, str):
            task = utils.Task(task, "")
        self.stack.append((task, Monitor(task=task.task)))

    def log_production(
        self,
        available_productions: List[Tuple[str, production.Production]],
        chosen_production: Optional[Tuple[str, production.Production]],
        time: int,
        **kwargs,
    ) -> List[str]:

        if chosen_production is not None:
            self.last_production = (time, chosen_production[1])

        self.state_hash = 0
        for v in kwargs.values():
            self.state_hash += abs(hash(v))
        return self.stack[-1][-1].update(
            self.state_hash, available_productions, chosen_production
        )

    def undo_production(self, time: int) -> Optional[production.Production]:

        if self.last_production is not None and self.last_production[0] == time:
            self.stack[-1][-1].update(self.state_hash, [], None)
            logging.debug("undo-ing affordance error")
            return self.last_production[1]

    def update(self, success: bool, time: int):

        monitor = self.stack[-1][-1]
        if success:
            monitor.update(Monitor.done_state, [], None)
            path = monitor.find_path(Monitor.done_state)
            logging.debug("updating utility for:")
            visited = set()
            for i, (_, (_, p)) in enumerate(reversed(path)):
                if p not in visited:
                    # visited.add(p)
                    p.update_success(i)
                    logging.debug("  {}".format(p.__class__.__name__))
                else:
                    logging.debug(
                        "  {} ignored b/c repetition".format(p.__class__.__name__)
                    )
        else:
            monitor.update(Monitor.quit_state, [], None)
        monitor.close()
        self.stack.pop()


class Agent:
    gpt4_setting = [
        "action",
        "production_english",
        "production_refine",
        "production_code",
    ]

    llm_setting = [
        "query_action",
        # "query_production",
        # "query_code",
        # "load_code",
        # "production_code_repeat",
    ]

    def __init__(
        self,
        name: str,
        env: domain.Env,
        prompt_dir: str = "prompts",
        production_dir: str = "productions",
        task_description_file: str = "task_description.json",
        log_dir: str = None,
        parent_log_dir: str = "logs",
        production_temperature: float = 0.1,
        **kwargs,
    ):

        self.name = name
        self.env = env
        self.production_temperature = production_temperature
        self.clock = 0
        self.test = False

        self.misc = SimpleNamespace()
        for k, v in kwargs.items():
            setattr(self.misc, k, v)

        self.prompt_dir = prompt_dir
        self.production_dir = production_dir
        self._load_prompts()
        if log_dir is None:
            self.log_dir = os.path.join(
                parent_log_dir, datetime.now().strftime("%m%d%y%H%M%S")
            )
        else:
            self.log_dir = os.path.join(parent_log_dir, log_dir)
            # if os.path.isdir(self.log_dir):
            #     shutil.rmtree(self.log_dir)

        logs = ["gpt", "agent", "checkpoint", "old_production"]
        for name in logs:
            log_path = os.path.join(self.log_dir, name)
            setattr(self, "log_dir_" + name, log_path)
            if not os.path.isdir(log_path):
                os.makedirs(log_path)
        self.env.set_log_dir(os.path.join(self.log_dir, "env"))

        self.task_description_file = task_description_file

        # declarative memory
        self.location_history = domain.LocationHistory()
        self.spatial_knowledge = domain.SpatialKnowledge(
            self.env.current_location, self.env.location_matrix
        )
        self.object_knowledge = domain.ObjectKnowledge(self.spatial_knowledge)
        self.extra_instructions: List[str] = []
        self.task_stack = TaskStack()
        self.previous_tasks = {}
        self.action_history = domain.ActionHistory()
        self.existing_plan = " None"

        self.task_descriptions = {}
        self._load_task_descriptions()

        # procedural memory
        self.procedures: List[str, Tuple[str, production.Production]] = {}
        self.procedures_modules: Dict[str, ModuleType] = {}
        self.procedure_names: Dict[str, int] = {}
        self._load_all_procedures()

        logging.info(
            "agent {} created, logging into {}".format(self.name, self.log_dir_agent)
        )

    @property
    def current_task(self) -> utils.Task:
        return self.task_stack.current_task

    @property
    def current_location(self) -> str:
        return self.spatial_knowledge.current_location

    @property
    def production_stats(self) -> Dict[str, Tuple[float, int]]:

        return {
            name: (rule.utility, rule.num_calls)
            for name, rule in self.procedures.values()
        }

    @production_stats.setter
    def production_stats(self, stats: Dict[str, Tuple[float, int]]):

        for name, rule in self.procedures.values():
            if name in stats.keys():
                utility, num_calls = stats[name]
                rule.utility = utility
                rule.num_calls = num_calls

    def __repr__(self) -> str:

        return "\n".join(
            [
                "=" * 60,
                "Agent: {}    @ time {:4d}".format(self.name, self.clock),
                "-" * 60,
                "Task:",
                "-----",
                pformat(self.task_stack, indent=2),
                "",
                "Procedurals:",
                "------------",
                pformat(self.procedures, indent=2),
                "",
                "Declaratives:",
                "-------------",
                "Location History:",
                str(self.location_history),
                "",
                "Spatial Knowledge:",
                str(self.spatial_knowledge),
                "",
                "Object Knowledge:",
                str(self.object_knowledge),
                "",
                "Extra Instrucitons:",
                pformat(self.extra_instructions, indent=2),
                "-" * 60,
                "History:",
                "--------",
                str(self.action_history),
                "=" * 60 + "\n",
            ]
        )

    def _load_prompts(self):

        mapping = {
            "action_system": "action/system_all_in_one.txt",
            "action_info_looped": "action/user_info_loop_all_in_one.txt",
            "production_english": "production/user_production_english.txt",
            "production_english_task": "production/user_production_english_w_task.txt",
            "production_refine": "production/user_production_refine.txt",
            "production_system": "production/system.txt",
            "production_code": "production/user_code.txt",
            "production_code_task": "production/user_code_w_task.txt",
            "production_code_repeat": "production/user_code_repeat.txt",
            "production_update": "production/user_code_update.txt",
        }
        self.prompts = SimpleNamespace()
        for key, filename in mapping.items():
            with open(os.path.join(self.prompt_dir, filename), "r") as fin:
                setattr(self.prompts, key, fin.read())

    def _load_all_procedures(self):

        for file_name in os.listdir(self.production_dir):
            if re.fullmatch(r"_[a-zA-Z_0-9]*[^_].py", file_name) is not None:
                self._load_production(file_name.replace(".py", ""))
        weight_file = os.path.join(self.production_dir, "production_weights.json")
        if os.path.isfile(weight_file):
            with open(weight_file, "r") as fin:
                self.production_stats = json.load(fin)

    def _load_production(
        self, file_name: str, load_to_agent: bool = True
    ) -> production.Production:

        module_path = "{}.{}".format(self.production_dir, file_name)
        importlib.invalidate_caches()
        if module_path in self.procedures_modules.keys():
            logging.debug("reloading {}".format(module_path))
            self.procedures_modules[module_path] = importlib.reload(
                self.procedures_modules[module_path]
            )
        else:
            logging.debug("loading {}".format(module_path))
            self.procedures_modules[module_path] = importlib.import_module(module_path)
        production_module = self.procedures_modules[module_path]

        # production_module = importlib.import_module(module_path)

        production_name = production_module.production_name
        new_production = getattr(production_module, production_name)()

        if load_to_agent:
            if module_path not in self.procedures.keys():
                production_name_prefix = "".join(
                    x for x in production_name if x.isalpha()
                )
                if production_name_prefix in self.procedure_names.keys():
                    self.procedure_names[production_name_prefix] += 1
                else:
                    self.procedure_names[production_name_prefix] = 1

            self.procedures[module_path] = (production_name, new_production)
            logging.info("load {} from {}".format(production_name, module_path))

        # del production_module

        return new_production

    def _load_task_descriptions(self, file_name: Optional[str] = None):

        if file_name is None:
            file_name = self.task_description_file
        if os.path.isfile(file_name):
            with open(file_name, "r") as fin:
                self.task_descriptions = json.load(fin)
        else:
            self.task_descriptions = {}

    def _get_subtasks(self) -> List[str]:

        all_tasks = {
            rule.target_task
            for name, rule in self.procedures.values()
            if not rule.is_disabled()
        }
        all_tasks = [x for x in all_tasks if x != ""]
        return sorted(all_tasks)

    def _get_production_descriptions(self, task: Optional[str] = None) -> List[str]:

        descriptions = []
        for _, rule in self.procedures.values():
            if task is None or rule.target_task == task:
                descriptions.append(rule.english_description)
        return descriptions

    def _update_info(self):

        observation = self.env.get_observation()
        self.location_history.add_location(self.env.current_location, self.clock)
        self.spatial_knowledge.update(self.env.current_location, observation)
        self.object_knowledge.update(observation)
        self.possible_actions = (
            [
                "attend to subtask: {} {}".format(
                    x,
                    "(Apply anytime. End condition: {})".format(
                        self.task_descriptions[x]
                    ),
                )
                for x in self._get_subtasks()
                if x in self.task_descriptions.keys()
                # if not self._match_subtask(self.current_task, x)
            ]
            + self.env.get_actions()
            + ["special action: 'done'", "special action: 'quit'"]
        )

    def _match_subtask(self, task_str: str, pattern_str: str) -> bool:

        variables = re.findall(r"(<[^<^>]+)", task_str)
        for name in variables:
            pattern_str = pattern_str.replace(name, "(.*)")
        pattern = re.compile(pattern_str)
        return re.fullmatch(pattern, task_str) is not None

    def save(self, save_to: Optional[str] = None):

        if save_to is None:
            save_to = os.path.join(
                self.log_dir_checkpoint, "agent_{:04}.pkl".format(self.clock)
            )
        memory_list = [
            "location_history",
            "spatial_knowledge",
            "object_knowledge",
            "extra_instructions",
            "task_stack",
            "action_history",
            "name",
            "clock",
            "gpt4_setting",
            "procedure_names",
            "production_stats",
            "previous_tasks",
        ]
        data = {
            "agent": {key: getattr(self, key) for key in memory_list},
            "env": {
                "floorplan": self.env.floorplan,
                "current_location": self.env.current_location,
                "object_states": self.env.get_objects(),
            },
        }
        with open(save_to, "wb") as fout:
            pickle.dump(data, fout)

    def save_weights(self, save_to: Optional[str] = None):

        if save_to is None:
            save_to = os.path.join(self.log_dir, "production_weights.json")
        with open(save_to, "w") as fout:
            json.dump(self.production_stats, fout, indent=2)

    def save_state_diagram(self, save_to: Optional[str] = None):

        if save_to is None:
            save_to = os.path.join(self.log_dir, "state_diagram.gv")
        Monitor.render(save_to)

    @staticmethod
    def load(load_from: str) -> Tuple["Agent", domain.Env]:

        with open(load_from, "rb") as fin:
            data = pickle.load(fin)
        env = domain.Env(floorplan=data["env"]["floorplan"])
        env.move_to(data["env"]["current_location"])
        env.set_objects(data["env"]["object_states"])
        agent = Agent("", env)
        for key, value in sorted(data["agent"].items()):
            setattr(agent, key, value)

        return agent, env

    def query_production_english(
        self,
        action: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:

        if action is None or messages is None:
            action, messages = self.query_action_preferences()

        if self.current_task.task_template == "":
            query = self.prompts.production_english.format(
                action=action,
                options="\n".join(
                    " * "
                    + (option[: option.index("(") - 1] if "(" in option else option)
                    for option in self.possible_actions
                    if "<" in option
                ),
            )
        else:
            query = self.prompts.production_english_task.format(
                action=action, target_task=self.current_task.task_template
            )
        messages.append({"role": "user", "content": query})
        production_english, messages = utils.GPTquery(
            messages,
            os.path.join(
                self.log_dir_gpt, "{:04d}_production_english".format(self.clock)
            ),
            return_both=True,
            gpt4="production_english" in self.gpt4_setting,
            long_context="production_english" not in self.gpt4_setting,
        )

        return production_english, messages

    def query_production_refine(
        self, messages: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, List[Dict[str, str]]]:

        if messages is None:
            _, messages = self.query_produdction_english()
        messages.append({"role": "user", "content": self.prompts.production_refine})

        production_english, messages = utils.GPTquery(
            messages,
            os.path.join(
                self.log_dir_gpt, "{:04d}_production_english_refine".format(self.clock)
            ),
            return_both=True,
            gpt4="production_refine" in self.gpt4_setting,
            long_context="production_refine" not in self.gpt4_setting,
        )

        return production_english, messages

    def query_production_code(
        self,
        description: Optional[str] = None,
        repeat_until_pass: bool = False,
        action_chosen: Optional[str] = None,
    ) -> str:

        if description is None:
            description, _ = self.query_production_english()

        if self.current_task.task_template == "":
            query = self.prompts.production_code.format(
                english_description=description,
                list_of_tasks="\n".join(" * " + x for x in self._get_subtasks()),
            )
        else:
            query = self.prompts.production_code_task.format(
                english_description=description,
                target_task=self.current_task.task_template,
            )
        messages = [
            {"role": "system", "content": self.prompts.production_system},
            {"role": "user", "content": query},
        ]
        response, messages = utils.GPTquery(
            messages,
            os.path.join(self.log_dir_gpt, "{:04d}_production_code".format(self.clock)),
            gpt4="production_code" in self.gpt4_setting,
            return_both=True,
        )

        repeats = 0
        passed = False
        while repeat_until_pass and repeats < 8:
            logging.debug("debugging precondition of the production")
            repeats += 1
            try:
                tmp_name = "test_{}".format(datetime.now().strftime("%m%d%y%H%M%S"))
                self.load_production_code(description, response, lower_name=tmp_name)
                tmp_production = self._load_production(tmp_name, False)
            except ValueError:
                break
            expected_return = '(True, "")'
            error_function = "precondition"
            try:
                passed, message = tmp_production.precondition(
                    self.current_task.task,
                    self.current_location,
                    self.previous_tasks,
                    self.spatial_knowledge,
                    self.object_knowledge,
                )
            except Exception as e:
                passed = False
                message = "{}: {}".format(type(e).__name__, e)
            else:
                if passed:
                    try:
                        tmp_production.apply()
                    except Exception as e:
                        passed = False
                        error_function = "effect"
                        expected_return = action_chosen
                        message = "{}: {}".format(type(e).__name__, e)
            if passed:
                break
            else:
                logging.debug("current implementation raises an error")
                res = None
                if res is not None:
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": self.prompts.production_code_repeat.format(
                            task=self.current_task,
                            current_location=self.current_location,
                            location_history=self.location_history,
                            spatial_knowledge=self.spatial_knowledge,
                            object_knowledge=self.object_knowledge,
                            previous_tasks=" None"
                            if self.previous_tasks == {}
                            else "\n".join(
                                " * {}: {}".format(*entry)
                                for entry in self.previous_tasks.items()
                            ),
                            error_function=error_function,
                            expected_return=expected_return,
                            code_message=message,
                        ),
                    },
                )
                try:
                    response, messages = utils.GPTquery(
                        messages,
                        os.path.join(
                            self.log_dir_gpt,
                            "{:04d}_production_code_repeat_{}".format(
                                self.clock, repeats
                            ),
                        ),
                        gpt4="production_code" in self.gpt4_setting,
                        long_context=repeats > 1,
                        return_both=True,
                    )
                except ValueError:
                    break
            del tmp_production

        for file_name in glob(os.path.join(self.production_dir, "test_*.py")):
            os.remove(file_name)

        return response if passed else None

    def query_production_update(
        self, module_str: str, new_description: str, explanation: str
    ):

        file_name = module_str.replace(".", "/") + ".py"
        with open(file_name, "r") as fin:
            code = fin.read()
        new_name = os.path.join(
            self.log_dir_old_production, os.path.basename(file_name)
        )
        suffix = ""
        i = 2
        while os.path.isfile(new_name.replace(".py", suffix + ".py")):
            suffix = str(i)
            i += 1
        logging.info(
            "moving old production from `{}` to `{}`".format(file_name, new_name)
        )
        os.rename(file_name, new_name)
        messages = [
            {"role": "system", "content": self.prompts.production_system},
            {
                "role": "user",
                "content": self.prompts.production_update.format(
                    code=code,
                    old_description=self.procedures[module_str][1].english_description,
                    new_description=new_description,
                    explanation=explanation,
                    list_of_tasks="\n".join(" * " + x for x in self._get_subtasks()),
                ),
            },
        ]
        response = utils.GPTquery(
            messages,
            os.path.join(
                self.log_dir_gpt, "{:04d}_production_update".format(self.clock)
            ),
            gpt4="production_code" in self.gpt4_setting,
        )
        match = re.search(r"```py\n(.*)```", response, re.DOTALL)
        if match is None:
            logging.error("production code cannot be parsed")
            raise ValueError
            # pdb.set_trace()
        else:
            code_str = match.group(1)

        with open(file_name, "w") as fout:
            fout.write(code_str)

    def load_production_code(
        self, description: str, response: str, lower_name: Optional[str] = None
    ):

        description_match = re.search(r"\[Generalized Rule\]\n([^\n]*)\n", description)
        if description_match is None:
            logging.warning("no rule header found, using the entire description")
            english_description = description
        else:
            english_description = description_match.group(1)
        match = re.search(r"```py\n(.*)```", response, re.DOTALL)
        if match is None:
            logging.error("production code cannot be parsed")
            pdb.set_trace()
        else:
            code_str = match.group(1)
        match = re.search(r"class ([a-zA-Z0-9_]*):", code_str)
        class_name = match.group(1)
        remaining = code_str[match.span()[1] :]
        if lower_name is None:
            if class_name in self.procedure_names.keys():
                class_name += str(self.procedure_names[class_name] + 1)
            lower_name = utils.name_lower(class_name)
            load_to_agent = True
        else:
            load_to_agent = False
        code_lines = code_str.split("\n")
        code_lines = [
            "# flake8: noqa",
            "from production import *",
            "",
            'production_name = "{}"'.format(class_name),
            "",
            "",
            "class {}(Production):".format(class_name),
            '    english_description = "{}"'.format(
                english_description.replace('"', "'")
            ),
        ]
        code_str = "\n".join(code_lines) + remaining

        # pdb.set_trace()
        with open(
            os.path.join(self.production_dir, "{}.py".format(lower_name)), "w"
        ) as fout:
            logging.info("writing new production rule into {}".format(lower_name))
            fout.write(code_str)

        if load_to_agent:
            self._load_production(lower_name, load_to_agent=True)

    def query_action_all_in_one_with_blacklist(
        self, blacklisted: List[str] = []
    ) -> Tuple[str, str, str, List[Dict[str, str]]]:

        blacklisted.append("attend to subtask: {}".format(self.current_task))
        messages = [
            {"role": "system", "content": self.prompts.action_system},
            {
                "role": "user",
                "content": self.prompts.action_info_looped.format(
                    task=self.current_task,
                    time=self.clock,
                    current_location=self.current_location,
                    location_history=self.location_history,
                    spatial_knowledge=self.spatial_knowledge,
                    object_knowledge=self.object_knowledge,
                    previous_tasks=" None"
                    if self.previous_tasks == {}
                    else "\n".join(
                        " * {}: {}".format(*entry)
                        for entry in self.previous_tasks.items()
                    ),
                    action_history=self.action_history.get_recent(),
                    plan=self.existing_plan,
                    extra_instructions=" None"
                    if self.extra_instructions == []
                    else "\n".join(" * " + line for line in self.extra_instructions),
                    possible_actions="\n".join(
                        " * " + action
                        for action in self.possible_actions
                        if action not in blacklisted
                    ),
                    discouraged_actions="\n".join(
                        " * " + action for action in blacklisted
                    ),
                ),
            },
        ]
        full_response, messages = utils.GPTquery(
            messages,
            os.path.join(self.log_dir_gpt, "{:04d}_action".format(self.clock)),
            return_both=True,
            gpt4="action" in self.gpt4_setting,
        )
        full_response += "\n\n[END]\n"
        action_selected = re.search(
            r'\[Option Suggestion\]\n"?([\s\S]*?)"?\n', full_response
        ).group(1)
        try:
            purpose = re.search(
                r"\[[Pp]urpose]\n"
                "The purpose of the suggested option is to ([\s\S]*?)\n",  # noqa
                full_response,
            ).group(1)
        except AttributeError:
            logging.warning("purpose cannot be parsed, setting to None")
            purpose = " None"
        try:
            plan = re.search(r"\[Plan]\n([\s\S]*?)\n\n", full_response).group(1)
        except AttributeError:
            logging.warning("plan cannot be parsed, setting to None")
            plan = " None"

        logging.debug(
            "got action: {} for the purpose of {}, \n and plan: {}".format(
                action_selected, purpose, plan
            )
        )

        return action_selected, "for the purpose of " + purpose, plan, messages

    def run(
        self,
        task: utils.Task,
        max_steps: int = -1,
        save_checkpoints: bool = False,
        test: bool = False,
    ):

        self.task_stack = TaskStack(task)
        self.test = test
        while self.step() and (max_steps < 0 or self.env.count < max_steps):
            if save_checkpoints:
                self.save()

    def step(self) -> bool:

        if len(self.task_stack) == 0:
            logging.warning("no task set, nothing to do")
            return False

        self.clock += 1
        logging.info("steping through time {:4d}".format(self.clock))

        self._update_info()

        retry = True
        production_blacklist: List[production.Production] = []
        action_blacklist: List[str] = []

        while retry:
            retry = False

            looped = []
            production_return = self.apply_productions(production_blacklist)
            if production_return is not None:
                (
                    action_selected,
                    applied_production,
                ) = production_return.chosen_production
                justification = applied_production.english_description
                looped = self.task_stack.log_production(
                    production_return.available_productions,
                    production_return.chosen_production,
                    self.clock,
                    current_location=self.current_location,
                    spatial_knowledge=self.spatial_knowledge,
                    object_knowledge=self.object_knowledge,
                )
                self.existing_plan = None

            if production_return is None or looped != []:

                messages = None
                if production_return is None:
                    logging.info("no production is applicable, querying LLM for action")
                    (
                        action_selected,
                        justification,
                        _,
                        messages,
                    ) = self.query_action_all_in_one_with_blacklist(
                        blacklisted=action_blacklist
                    )
                elif looped != []:
                    logging.info(
                        "loop detected in production application, "
                        "querying LLM for alternative action"
                    )
                    (
                        action_selected_llm,
                        justification_llm,
                        _,
                        messages,
                    ) = self.query_action_all_in_one_with_blacklist(
                        looped + action_blacklist
                    )
                    if action_selected_llm is not None:
                        logging.debug(
                            "changing action from {} to {}".format(
                                action_selected, action_selected_llm
                            )
                        )
                        action_selected = action_selected_llm
                        justification = justification_llm

                if "query_production" in self.llm_setting and messages is not None:
                    messages = utils.clean_formatting(messages)
                    production_english, messages = self.query_production_english(
                        action_selected, messages
                    )
                    if "query_production_refine" in self.llm_setting:
                        production_english, messages = self.query_production_refine(
                            messages
                        )
                else:
                    production_english = None
                if "query_code" in self.llm_setting and production_english is not None:
                    # pdb.set_trace()
                    code_response = self.query_production_code(
                        production_english,
                        repeat_until_pass="production_code_repeat" in self.llm_setting,
                    )
                else:
                    code_response = None
                if "load_code" in self.llm_setting and code_response is not None:
                    self.load_production_code(production_english, code_response)

            match = re.fullmatch(
                '"?(special action|motor action|attend to sub-?task): ?(.*)"?',
                action_selected.lower(),
            )
            if match is None:
                logging.error("action cannot be parsed: {}".format(action_selected))
                pdb.set_trace()
                continue
            else:
                action_type, action_str = match.groups()
            # action_result = None

            if action_type == "special action":
                if "done" in action_str:
                    logging.info(
                        "the current task: {} is completed".format(self.current_task)
                    )
                    self.existing_plan = " None"
                    self.previous_tasks[self.current_task.task] = True
                    self.task_stack.update(True, self.clock)
                    if self.test and len(self.task_stack) == 0:
                        raise utils.Done
                elif "quit" in action_str:
                    logging.info(
                        "the current task: {} is impossible".format(self.current_task)
                    )
                    self.existing_plan = " None"
                    self.previous_tasks[self.current_task.task] = False
                    self.task_stack.update(False, self.clock)
                    if self.test and len(self.task_stack) == 0:
                        raise utils.Quit
                else:
                    logging.error("unrecognized special action: {}".format(action_str))
                    pdb.set_trace()
            elif action_type == "motor action":
                logging.info("executing {}".format(action_str.strip()))
                try:
                    self.env.perform_action(action_str.strip())
                except utils.AffordanceError as e:
                    logging.debug("got affordance error: {}".format(e))
                    retry = True
                    action_blacklist.append("{}: {}".format(action_type, action_str))
                    production_applied = self.task_stack.undo_production(self.clock)
                    if production_applied is not None:
                        logging.debug("blacklisted {}".format(production_applied))
                        production_blacklist.append(production_applied)
                    else:
                        logging.error("LLM action yields affordance error")
                self._update_info()
            else:
                self.task_stack.append(action_str.strip().lower())
                self.existing_plan = " None"
                logging.info(
                    "attending to new task, current task stack: \n{}".format(
                        self.task_stack
                    )
                )

            self.action_history.add_action(
                action_selected, justification, None, time=self.clock
            )

        with open(
            os.path.join(self.log_dir_agent, "{:04d}_dump.txt".format(self.clock)), "w"
        ) as fout:
            fout.write(str(self))
        # pdb.set_trace()

        return True

    def apply_productions(
        self, production_blacklist: List[production.Production] = []
    ) -> Optional[production.Production]:

        applicables: List[production.Production] = []
        for name, rule in sorted(self.procedures.values()):
            if rule in production_blacklist or rule.is_disabled():
                logging.debug("{} is blacklisted".format(name))
            else:
                try:
                    res, message = rule.precondition(
                        self.current_task.task,
                        self.current_location,
                        self.previous_tasks,
                        self.spatial_knowledge,
                        self.object_knowledge,
                    )
                    if res:
                        rule.apply()
                except Exception as e:
                    logging.error(
                        "The following exception occured when testing for {}:\n"
                        "{}: {}".format(name, type(e).__name__, e)
                    )
                    res = False
                    message = "{}: {}".format(type(e).__name__, e)
                    # pdb.set_trace()
                if res:
                    logging.debug("{} applicable".format(name))
                    applicables.append((name, rule))
                else:
                    logging.debug("{} not applicable because {}".format(name, message))

        if applicables == []:
            logging.debug("no production applicable")
            res = None
            pdb.set_trace()
        else:
            weights = [
                math.exp(
                    (1 if p.num_calls == 0 else p.utility)
                    / self.production_temperature
                    * (2 if self.test else 1)
                )
                for name, p in applicables
            ]
            name, selected_production = random.choices(
                applicables, weights=weights, k=1
            )[0]
            logging.debug(
                "applicable productions: \n{}".format(pformat(applicables, indent=2))
            )
            logging.debug("production weights: \n{}".format(pformat(weights, indent=2)))
            logging.info("production: {} is selected".format(selected_production))
            res = SimpleNamespace(
                available_productions=[(p.apply(), p) for name, p in applicables],
                chosen_production=(selected_production.apply(), selected_production),
            )

        return res


class AgentActionOnly(Agent):
    # gpt4_setting = []
    llm_setting = ["query_action"]

    def _load_prompts(self):

        mapping = {
            "action_system": "action/system_no_extra.txt",
            "action_info_looped": "action/user_info_no_extra.txt",
        }
        self.prompts = SimpleNamespace()
        for key, filename in mapping.items():
            with open(os.path.join(self.prompt_dir, filename), "r") as fin:
                setattr(self.prompts, key, fin.read())

    def _load_all_procedures(self):
        pass

    def _load_task_descriptions(self):
        self.task_descriptions = {}

    def _get_subtasks(self):
        return []

    def step(self) -> bool:

        self.clock += 1
        logging.info("steping through time {:4d}".format(self.clock))
        self._update_info()

        retry = 0
        blacklist = []
        while retry < 3:
            (
                action_selected,
                justification,
                _,
                messages,
            ) = self.query_action_all_in_one_with_blacklist()

            self.action_history.add_action(
                action_selected, justification, None, self.clock
            )

            match = re.fullmatch(
                '"?(special action|motor action|attend to sub-?task): ?(.*)"?',
                action_selected,
            )
            if match is None:
                logging.error("action cannot be parsed: {}".format(action_selected))
                retry += 1
                continue
            else:
                action_type, action_str = match.groups()
                action = "{}: {}".format(action_type, action_str)

            if action_type == "special action":
                if "done" in action_str:
                    if self.test:
                        raise utils.Done
                    else:
                        return True
                elif "quit" in action_str:
                    if self.test:
                        raise utils.Quit
                    else:
                        return True
                else:
                    logging.error("unrecognized special action: {}".format(action_str))
                    blacklist.append(action)
                    retry += 1
                    continue
            elif action_type == "motor action":
                logging.info("executing {}".format(action_str.strip()))
                try:
                    self.env.perform_action(action_str.strip())
                except utils.AffordanceError:
                    blacklist.append(action)
                    retry += 1
                    continue
                else:
                    return True
            else:
                retry += 1
                continue

        return False
