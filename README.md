**This directory is currently identical to what was submitted to the review. Will clean up later.**


Setup
----
1. (Optional) set up an [conda virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
2. Install dependencies:
```sh
python3 -m pip install ai2thor==4.3.0 fire openai pandas matplotlib graphviz
```
3. Setup [OpenAI API key](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety#h_a1ab3ba7b2)

File Structure
----
```sh
code_and_data/
 |--agent.py                   # agent, task stack, loop detector
 |--domain.py                  # domain specific classes, e.g., object, receptacle, etc.
 |--env/                       # wrapper for AI2THOR environment
 |experiment.xlsx              # the spreadsheet and t-test results of the experiments
 |--experiments.py             # entry point for experiments
 |--locations/                 # initialization for the simulator
 |--logs/
   |--bootstrapping/           # logs of the bootstrapping process
     |--critics/               # output of critics
     |--[task]_[trial]/        # logs of bootstrapping a single task instance
       |--env/                 # screenshots of the environment during the agent's task execution
       |--gpt/                 # logs of GPT4 responses (if any)
     |--task_descriptions.json # end conditions for each tasks summarized by the critic
   |--test_agent/              # logs of the testing process for our bootstrapped agent
     |--[task]_[trial]         # log of one task instance, same format as above
   |--test_baseline/           # logs of the testing process for the baseline agent, same format as above
                               # due to file upload size constraint on CMT, some of the images are omitted
 |--oracle_database.json       # all the world knowledge bootstrapped
 |--production.py              # the production rule template and OracleQuery definition
 |--productions/
   |--_*.py                    # the production rules generated during the bootstrapping process
   |--production_weights.json  # the learned weights of the production rules
 |--prompts/                   # the prompt templates used for GPT4
 |--tokens.jpg                 # figure 7 in the paper
 |--utils.py                   # some helper functions
```

Bootstrapping
----
1. (Optional) Rename `productions` to something else or delete it completely, otherwise the productions will be reused
2. Run `python3 experiments.py train_agent`
3. Check `logs/training_log` and `productions` for the bootstrapping process

Notes:
* This process costs around $40
* As explained in the technically appendix, the bootstrapped productions may not be the same as the ones we provided
* During the execution process the environment will throw A LOT of warnings, that is expected (same for the testing below)

Testing
----
* For the baseline, run `python3 experiments.py test_agent "AgentActionOnly"`\
The results should be in `logs/AgentActionOnly`\
This process costs around $120

* For the bootstrapped agent, run `python3 experiments.py test_agent "Agent"`\
The results should be in `logs/Agent`\
This process costs round $5

* Each test will spawn a new simulator environment, if that is a problem, try running each individual test one at a time
