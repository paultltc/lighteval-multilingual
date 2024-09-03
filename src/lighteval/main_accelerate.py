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

import os
from datetime import timedelta

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.hierarchical_logger import hlog_warn, htrack, htrack_block
from lighteval.models.model_config import create_model_config
from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available, is_tgi_available
from lighteval.utils.utils import build_config_from_args

from nanotron.config import Config, LightEvalConfig, get_config_from_file

if not is_accelerate_available() and not is_tgi_available():
    hlog_warn("Using either accelerate or text-generation to run this script is advised.")

HF_TOKEN = os.getenv("HF_TOKEN")

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None


@htrack()
def main(args):
    env_config = EnvConfig(token=HF_TOKEN, cache_dir=args.cache_dir)

    with htrack_block("Load configs"):
        if args.lighteval_override is None:
            lighteval_config = build_config_from_args(args)
        else:
            lighteval_config = get_config_from_file(args.lighteval_override, config_class=LightEvalConfig)

    evaluation_tracker = EvaluationTracker(
        output_dir=lighteval_config.logging.local_output_path,
        hub_results_org=lighteval_config.logging.hub_repo_tensorboard,
        # push_results_to_hub=lighteval_config.logging.push_results_to_hub,
        # push_details_to_hub=lighteval_config.logging.push_details_to_hub,
        # push_results_to_tensorboard=lighteval_config.logging.push_results_to_tensorboard,
        tensorboard_metric_prefix=lighteval_config.logging.tensorboard_metric_prefix,
        token=HF_TOKEN,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=env_config,
        job_id=os.environ.get("SLURM_JOB_ID", None),
        dataset_loading_processes=lighteval_config.tasks.dataset_loading_processes,
        custom_tasks_directory=lighteval_config.tasks.custom_tasks,
        override_batch_size=lighteval_config.batch_size,
        num_fewshot_seeds=lighteval_config.tasks.num_fewshot_seeds,
        max_samples=lighteval_config.tasks.max_samples,
        use_chat_template=False,
        system_prompt=None,
    )

    model_config = create_model_config(
        use_chat_template=args.use_chat_template,
        override_batch_size=args.override_batch_size,
        model_args=args.model_args,
        model_config_path=args.model_config_path,
        accelerator=accelerator,
    )

    pipeline = Pipeline(
        tasks=lighteval_config.tasks.tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()

    pipeline.show_results()

    results = pipeline.get_results()

    pipeline.save_and_push_results()

    return results
