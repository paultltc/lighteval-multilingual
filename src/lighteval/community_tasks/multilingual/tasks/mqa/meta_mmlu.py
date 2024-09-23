from typing import Literal, get_args

from lighteval.community_tasks.multilingual.tasks.utils.prompts import get_meta_mmlu_prompt
from .mlmm import MMLU_SUBSET
from lighteval.metrics.metrics import Metrics 

from lighteval.tasks.lighteval_task import LightevalTaskConfig

LANGS = Literal["es", "de", "es", "fr", "hi", "it", "pt", "th"]

class MetaMMLUTask(LightevalTaskConfig):
    NAME = "meta_mmlu"
    LANGS = Literal["es", "de", "es", "fr", "hi", "it", "pt", "th"]
    SUBSETS = MMLU_SUBSET

    def __init__(self, lang: LANGS, task: SUBSETS):
        subset = "Meta-Llama-3.1-8B-Instruct-evals__multilingual_mmlu__details" if lang == 'en' \
                    else f"Meta-Llama-3.1-8B-Instruct-evals__multilingual_mmlu_{lang}__details"
        super().__init__(
            name=f"meta_mmlu-{lang}:{task}",
            prompt_function=get_meta_mmlu_prompt(lang),
            suite=("custom",),
            hf_repo="meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
            hf_subset=subset,
            filter=lambda line: line["subtask_name"] == f"mmlu_{lang}_chat.{task}",
            evaluation_splits=("latest",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_pmi, 
                Metrics.loglikelihood_prob, 
                Metrics.loglikelihood_prob_norm, 
                Metrics.loglikelihood_prob_norm_token, 
                Metrics.loglikelihood_prob_norm_pmi,
            ),
        )

def get_meta_mmlu_tasks(lang: LANGS):
    return [MetaMMLUTask(lang, subset) for subset in get_args(MMLU_SUBSET)]