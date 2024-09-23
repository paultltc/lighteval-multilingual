from functools import partial
from typing import Literal, get_args

from ..utils.prompts import get_m_truthfulqa_prompt, get_truthfulqa_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

class M_TruthfulQATask(LightevalTaskConfig):
    NAME = "m_truthfulqa"
    LANGS = Literal["ar", "bn", "ca", "da", "de", "es", "eu", "fr", "gu", "hi", "hr", "hu", "hy", "id", "it", "kn", "ml", "mr", "ne", "nl", "pt", "ro", "ru", "sk", "sr", "sv", "ta", "te", "uk", "vi", "zh"]
    TYPES = Literal["mc1", "mc2"]

    def __init__(self, lang: LANGS, type: TYPES = "mc1"):
        super().__init__(
            name=f"m_truthfulqa-{lang}",
            prompt_function=get_m_truthfulqa_prompt(lang, type),
            suite=("custom",),
            hf_repo="alexandrainst/m_truthfulqa",
            hf_subset=lang,
            trust_dataset=True,
            evaluation_splits=("val",),
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

# ----------------------------------- OKAPI ---------------------------------- #
# class M_TruthfulQATask(LightevalTaskConfig):
#     NAME = "m_truthfulqa"
#     LANGS = Literal["ar", "bn", "ca", "da", "de", "es", "eu", "fr", "gu", "hi", "hr", "hu", "hy", "id", "it", "kn", "ml", "mr", "ne", "nl", "pt", "ro", "ru", "sk", "sr", "sv", "ta", "te", "uk", "vi", "zh"]

#     def __init__(self, lang: LANGS, type: Literal["mc1", "mc2"] = "mc1"):
#         super().__init__(
#             name=f"truthfulqa-{lang}:{type}",
#             prompt_function=get_truthfulqa_prompt(lang, type),
#             suite=("custom",),
#             hf_repo="jon-tow/okapi_truthfulqa",
#             hf_subset=lang,
#             hf_revision="cdd5db1a66fd04105622109d1c2a5cbc8cde7586",
#             trust_dataset=True,
#             evaluation_splits=("validation",),
#             metric=(
#                 Metrics.loglikelihood_acc,
#                 Metrics.loglikelihood_acc_norm_nospace,
#                 Metrics.loglikelihood_acc_norm_token,
#                 Metrics.loglikelihood_acc_norm_pmi,
#                 Metrics.loglikelihood_prob,
#                 Metrics.loglikelihood_prob_norm,
#                 Metrics.loglikelihood_prob_norm_token,
#                 Metrics.loglikelihood_prob_norm_pmi,
#             ),
#         )
        