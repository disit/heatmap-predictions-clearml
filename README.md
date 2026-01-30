# Heatmap Predictions Service (ClearML Backend)

This repository contains the Python backend logic for the **Heatmap Generation Service**. It is designed to run as a **ClearML Task**, using the **API Sporadic** method.

## Project Structure

```text
heatmap-predictions-clearml/
│
├── logic/                          # Core application logic
│   ├── config.py                   # Configuration parameters
│   ├── helper.py                   # General utility functions
│   └── heatmap/                    # Heatmap specific modules
│       ├── heatmap.py              # Logic main unit, orchestrates the other modules
│       ├── data_retrieval.py       # Handles data retrieval from sensors in the area
│       ├── data_preprocessing.py   # Handles all operations needed on data before interpolation
│       ├── data_interpolation.py   # Handles grid creation and interpolation
│       └── data_upload.py          # Handles file upload to storage
│
├── tasks/                  # ClearML Entrypoints
│   └── task_heatmap.py     # Main script to initialize/execute the task
│
├── requirements.txt        # Python dependencies
└── README.md
```

## How to use

The code present in this repo will be passed to the task that will be used as a template in the Sporadic method. To pass the code to the task we need to specify some commands:


### Task creation


```text

task = Task.create(
    project_name="Heatmap",
    task_name="heatmap_template_task",
    task_type=Task.TaskTypes.application
)

```

### (OPTIONAL) Loading configuration

During the template task creation we may use a config.py file (a local version that might differ from the one in the repo) to connect the configuration to the task.
This operation lets us use a dynamic configuration that will override the repo one and might also be modified between each execution.

```text

import config

config_dict = {}
for k, v in vars(config).items():
    if k.startswith('__'):
        continue
    
    # If the value is a set (es. PHYSICAL_POSITIVE_CATS), we convert it into a list
    if isinstance(v, set):
        config_dict[k] = list(v)
    else:
        config_dict[k] = v

print(f"Config loading: {len(config_dict)} variables found.")

task.set_configuration_object(
    name="General",       # Nome della sezione nella UI
    config_dict=config_dict, 
    description="Config generated from logic/config.py"
)

```

### Code injection

This is the most important part of the script.

We need to tell the task where to find the code.

```text

task.set_repo(
    repo="https://github.com/disit/heatmap-predictions-clearml",
    branch="main",
    commit=""
)

```

and then we need to tell it what file is the entrypoint.

```text

task.set_script(
    working_dir=".",
    entry_point="tasks/task_heatmap.py"
)

```

### Task ID

After the task is created, we'll need to put the id of the task into the API manager, connected to the endpoint.
From there everytime we make a request to that, a task will be cloned from the template, the one we created, the params, passed through the request, will be connected to the new task and then it will be executed.