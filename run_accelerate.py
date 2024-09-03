# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Example run command:
accelerate config
accelerate launch run_evals_accelerate.py --tasks="leaderboard|hellaswag|5|1" --output_dir "/scratch/evals" --model_args "pretrained=gpt2"
"""

import argparse

import os

from lighteval.main_accelerate import main


def get_parser():
    parser = argparse.ArgumentParser()
    task_type_group = parser.add_mutually_exclusive_group(required=True)

    # Model type: either use a config file or simply the model name
    task_type_group.add_argument("--model-config-path")
    task_type_group.add_argument("--model-args")

    # Lighteval config
    parser.add_argument(
        "--lighteval-override",
        type=str,
        help="Path to an optional YAML or python Lighteval config to override part of the checkpoint Lighteval config",
    )

    # Debug
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--override_batch_size", type=int, default=-1)
    parser.add_argument("--job_id", type=str, help="Optional Job ID for future reference", default="")
    # Saving
    parser.add_argument("--logging_dir", type=str, help="Base dir for saving logs (e.g., './output' or 'hf://repo' or 's3://bucket/prefix')", default="./output")
    parser.add_argument("--save_results", action="store_true", help="Save results to the logging dir", default=True)
    parser.add_argument("--save_details", action="store_true", help="Save details to the logging dir")
    parser.add_argument("--save_to_tensorboard", action="store_true", help="Save tensorboard logs to the logging dir")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.getenv("HF_HOME"),
        help="Cache directory for downloaded datasets & model, defaults to `HF_HOME` environment variable",
    )
    parser.add_argument("--use_chat_template", default=False, action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--dataset_loading_processes", type=int, default=1)
    parser.add_argument(
        "--custom_tasks",
        type=str,
        default=None,
        help="Path to a file with custom tasks (a TASK list of dict and potentially prompt formatting functions)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated ids of tasks, e.g. 'original|mmlu:abstract_algebra|5' or path to a text file with a list of tasks",
    )
    parser.add_argument("--num_fewshot_seeds", type=int, default=1, help="Number of trials the few shots")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, unknowns = parser.parse_known_args()
    main(args)
