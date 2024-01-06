import json
import logging
import os
import pdb  # noqa
import re
from glob import glob
from pprint import pformat
from time import sleep
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import openai

from env.env import AffordanceError, gripper_name, gripper_type  # noqa


class Quit(Exception):
    pass


class Done(Exception):
    pass


class Task(NamedTuple):
    task: str
    task_template: str

    def __repr__(self) -> str:
        return self.task


def GPTquery(
    messages: List[Dict[str, str]],
    log: Optional[str] = None,
    verification_func: Callable[[str], bool] = lambda _: True,
    strip_func: Callable[[str], str] = lambda s: s,
    long_context: bool = False,
    gpt4: bool = False,
    max_tries: int = 1,
    return_both: bool = False,
    temperature: float = 0,
) -> Union[str, List[Dict[str, str]]]:

    response = None
    if gpt4:
        # model = "gpt-4-32k-0613" if long_context else "gpt-4-0613"
        model = "gpt-4-0613"
    else:
        model = "gpt-3.5-turbo-16k-0613"  # if long_context else "gpt-3.5-turbo-0613"
    logging.debug("sending the following message to openai: {}".format(model))
    logging.debug(pformat(messages, indent=2))

    res = None
    # pdb.set_trace()
    if res is not None:
        return res

    count = 0
    while True:
        count += 1
        try:
            full_response = openai.ChatCompletion.create(
                model=model, messages=messages, temperature=temperature
            )
            sleep(1)
        except openai.error.RateLimitError:
            logging.warning(
                "failured for {} time(s) b/c openai rate limit".format(count)
            )
            sleep(2 ** count)
            max_tries += 1
        except (
            openai.error.ServiceUnavailableError,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.Timeout,
        ):
            logging.warning("openai service unavailable")
            sleep(2 ** count)
            max_tries += 1
        except openai.error.InvalidRequestError as e:
            logging.warning("got {}: {}".format(type(e), e))
            meta = {"finish_reason": "length"}
            raise ValueError
        else:
            response = full_response["choices"][0]["message"]["content"]
            meta = {
                "finish_reason": full_response["choices"][0]["finish_reason"],
                "total_tokens": full_response["usage"]["total_tokens"],
                "model": model,
            }
            if verification_func(response):
                break
            else:
                logging.info("response failed verification test, retrying")
            if count >= max_tries:
                if response is None:
                    logging.error("maximum tries exhausted, still no response")
                    response = ""
                    # pdb.set_trace()
                else:
                    logging.warn(
                        "maximum tries exhausted, "
                        "no valid response, passing the latest response"
                    )
                break

    if meta["finish_reason"] == "length":
        if not gpt4 and not long_context:
            logging.info("response too long, using longer context")
            return GPTquery(
                messages=messages,
                log=log,
                verification_func=verification_func,
                strip_func=strip_func,
                long_context=True,
                gpt4=gpt4,
                max_tries=max_tries,
                return_both=return_both,
                temperature=temperature,
            )
        else:
            raise ValueError

    all_messages = messages + [{"role": "assistant", "content": response}]

    if log is not None:
        if log.endswith(".json"):
            with open(log, "w") as fout:
                json.dump(all_messages, fout, indent=2)
        elif log.endswith(".txt"):
            with open(log, "w") as fout:
                for message in all_messages:
                    fout.write("-" * (2 + len(message["role"])) + "\n")
                    fout.write(" {}\n".format(message["role"].upper()))
                    fout.write("-" * (2 + len(message["role"])) + "\n\n")
                    fout.write(message["content"])
                    fout.write("\n" + "=" * 80 + "\n")
                fout.write("\n<<<META INFO>>>\n")
                fout.write(
                    "total tokens: {total_tokens} "
                    "finished reason: {finish_reason} "
                    "model name: {model}".format(**meta)
                )
        else:
            with open(log + ".json", "w") as fout:
                json.dump(all_messages, fout, indent=2)
            with open(log + ".txt", "w") as fout:
                for message in all_messages:
                    fout.write("-" * (2 + len(message["role"])) + "\n")
                    fout.write(" {}\n".format(message["role"].upper()))
                    fout.write("-" * (2 + len(message["role"])) + "\n\n")
                    fout.write(message["content"])
                    fout.write("\n" + "=" * 80 + "\n")
                fout.write("\n<<<META INFO>>>\n")
                fout.write(
                    "total tokens: {total_tokens} "
                    "finished reason: {finish_reason} "
                    "model name: {model}".format(**meta)
                )

    response = strip_func(response)
    return (response, all_messages) if return_both else response


def clean_formatting(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:

    new_messages = []
    for entry in messages:
        if entry["role"] != "user":
            new_messages.append(entry)
        else:
            match = re.search(
                r"[Uu]se the following format(.*)\Z", entry["content"], re.DOTALL
            )
            if match is not None:
                new_messages.append(
                    {"role": "user", "content": entry["content"][: match.span()[0]]}
                )
            else:
                new_messages.append(entry)
    return new_messages


def name_lower(name: str) -> str:

    return "".join("_" + x.lower() if x.isupper() else x for x in name.strip())


def name_split(name: str) -> Set[str]:

    ignores = ["in front of", "sliced", "cooked"]
    name = name.strip().lower()
    for kw in ignores:
        name = name.replace(kw, "")
    name_alph = "".join(x if x.isalnum() else "_" for x in name)
    return {x for x in name_alph.split("_") if x not in {"", "a", "an"}}


def set_logging(level: str = "INFO"):
    logging.basicConfig(
        force=True,
        level=level,
        format="%(asctime)s [%(levelname)s] "
        "(%(module)s:%(lineno)d @%(process)d) %(message)s",
    )


def count(log_dir) -> Tuple[int, int]:

    log_files = glob(os.path.join(log_dir, "gpt", "*.txt"))
    tokens = 0
    for log_file in log_files:
        with open(log_file) as fin:
            text = fin.read()
            token_str = re.search(r"total tokens: ([0-9]+) ", text).group(1)
            tokens += int(token_str)
    env_files = glob(os.path.join(log_dir, "env", "*.jpg"))
    return len(env_files), tokens


def draw_train_tokens(log_dir: str):

    import pandas as pd
    from matplotlib import pyplot as plt

    plt.figure()
    plt.rcParams.update({"font.size": 16})

    counts = {
        "Action": [],
        "Production English": [],
        "Production Code": [],
        "Code Revision": [],
    }
    for prefix in reversed(["go", "find", "pick", "slice", "put"]):
        for suffix, category in [
            ("*_action.txt", "Action"),
            ("*_production_english.txt", "Production English"),
            ("*_production_code.txt", "Production Code"),
            ("*_production_code_repeat_*.txt", "Code Revision"),
        ]:
            total = 0
            for file_name in glob(
                os.path.join(log_dir, "{}*".format(prefix), "gpt", suffix)
            ):
                with open(file_name) as fin:
                    text = fin.read()
                    token_str = re.search(r"total tokens: ([0-9]+) ", text).group(1)
                    total += int(token_str)
            counts[category].append(total)

    df = pd.DataFrame(
        counts,
        index=reversed(
            ["explore", "find", "pick and place", "slice", "put things away"]
        ),
    )
    df.plot.barh(stacked=True, figsize=(15, 5)).legend(bbox_to_anchor=(1.0, 1.0))
    plt.ylabel("task")
    plt.xlabel("number of tokens used")
    plt.tight_layout()
    plt.savefig("tokens.jpg")
