


from typing import Literal
from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_c3_prompt, get_cmnli_prompt, get_mlqa_prompt, get_ocnli_prompt, get_xnli_prompt
from lighteval.metrics.metrics import Metrics
from ..utils.translation_literals import FULL_STOP
from lighteval.tasks.lighteval_task import LightevalTaskConfig


class CMRC2018Task(LightevalTaskConfig):
    def __init__(self, max_generation_chars: int | None = None):
        super().__init__(
            name=f"cmrc",
            prompt_function=get_mlqa_prompt("zh"),
            suite=("custom",),
            hf_repo="clue/clue",
            hf_subset="cmrc2018",
            evaluation_splits=("trial",),
            filter=lambda x: all(len(ans) < max_generation_chars for ans in x["answers"]["text"]) if max_generation_chars else True,
            few_shots_split="train",
            generation_size=100,
            metric=(get_qa_metric("zh", "exact"), get_qa_metric("zh", "f1")),
            stop_sequence=("\n",),
        )

class C3Task(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"c3",
            prompt_function=get_c3_prompt("zh"),
            suite=("custom",),
            hf_repo="clue/clue",
            hf_subset="c3",
            evaluation_splits=("validation",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi,
            ),
        )
        
class OCNLI(LightevalTaskConfig):
    def __init__(self, version: Literal[1,2]):
        super().__init__(
            name=f"ocnli-bool{f'-v{version}' if version != 1 else ''}-zh",
            prompt_function=get_ocnli_prompt("zh", version),
            suite=("custom",),
            hf_repo="clue/clue",
            hf_subset="ocnli",
            # Only keep the positive and negative examples
            filter=lambda x: int(x["label"]) in [1, 2],
            evaluation_splits=("validation",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.loglikelihood_prob,
                Metrics.loglikelihood_prob_norm,
                Metrics.loglikelihood_prob_norm_token,
                Metrics.loglikelihood_prob_norm_pmi,
            ),
        )
    
    
class CMNLITask(LightevalTaskConfig):
    def __init__(self, version: Literal[1,2]):
        super().__init__(
            name=f"cmnli-bool{f'-v{version}' if version != 1 else ''}-zh",
            prompt_function=get_cmnli_prompt("zh", version),
            suite=("custom",),
            hf_repo="fenffef/cmnli",
            hf_subset="default",
            # Only keep the positive and negative examples
            filter=lambda x: x["label"] in ["entailment", "contradiction"] and x["sentence1"].endswith(FULL_STOP["zh"]),
            evaluation_splits=("validation",),
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.loglikelihood_prob,
                Metrics.loglikelihood_prob_norm,
                Metrics.loglikelihood_prob_norm_token,
                Metrics.loglikelihood_prob_norm_pmi,
            ),
        )


# TODO: DEEPSEEK Also uses CHID + CCPM, but I am not sure how to prompt it