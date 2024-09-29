from typing import Literal

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

from ..utils.prompts import get_3mlu_prompt

from lighteval.community_tasks.multilingual.tasks.utils.translation_literals import LANG_NAMES_INVERTED

class M3LU_TASK(LightevalTaskConfig):
    NAME = "3mlu"
    LANGS = Literal['ur', 'hr', 'pt', 'vi', 'fi', 'fr', 'eu', 'de', 'ar', 'hi', 'es', 'ko', 'ta', 'te', 'pl', 'sr', 'tr', 'hu', 'id', 'bg', 'zh', 'et', 'el', 'lt', 'it', 'bn', 'ru', 'sq', 'ja']

    def __init__(self, lang: LANGS):
        subset = LANG_NAMES_INVERTED[lang]
        super().__init__(
            name=f"xcopa-{lang}",
            suite=("custom",),
            prompt_function=get_3mlu_prompt(lang),
            hf_repo="paultltc/3mlu",
            hf_subset=subset,
            evaluation_splits=("test",),
            few_shots_split="val",
            few_shots_select=None,
            generation_size=-1,
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
            stop_sequence=("\n",),
            trust_dataset=True,
            version=0,
        )
