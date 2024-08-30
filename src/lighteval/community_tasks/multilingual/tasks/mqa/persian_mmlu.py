from typing import Literal

from ..utils.prompts import get_khayyam_challenge_prompt
from ..utils.translation_literals import LANG_NAMES_INVERTED
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

class PersianMMLU(LightevalTaskConfig):
    NAME = "m_truthfulqa"
    LANGS = Literal['fa']

    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"mmlu-{lang}",
            prompt_function=get_khayyam_challenge_prompt(lang),
            suite=("custom",),
            hf_repo="raia-center/khayyam-challenge",
            hf_subset="default",
            trust_dataset=True,
            evaluation_splits=("train",),
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