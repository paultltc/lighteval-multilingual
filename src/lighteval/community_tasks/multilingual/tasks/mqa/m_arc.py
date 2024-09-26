from typing import Literal

from ..utils.prompts import get_arc_prompt, get_m_arc_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

class M_ARCTask(LightevalTaskConfig):
    NAME = "m_arc"
    LANGS =  Literal["ar", "bn", "ca", "da", "de", "en", "es", "eu", "fr", "gu", "hi", "hr", "hu", "hy", "id", "is", "it", "kn", "ml", "mr", "nb", "ne", "nl", "pt", "ro", "ru", "sk", "sr", "sv", "ta", "te", "uk", "vi", "zh"]  

    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"m_arc-{lang}",
            prompt_function=get_m_arc_prompt(lang),
            suite=("custom",),
            hf_repo="alexandrainst/m_arc",
            hf_subset=lang,
            hf_revision="823d5d7bfaf8974a3ab52a825b6cf4903b35dbc4",
            trust_dataset=True,
            evaluation_splits=("test",),
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

# class M_ARCTask(LightevalTaskConfig):
#     NAME = "m_arc"
#     LANGS =  Literal["ar", "bn", "ca", "da", "de", "es", "eu", "fr", "gu", "hi", "hr", "hu", "hy", "id", "it", "kn", "ml", "mr", "ne", "nl", "pt", "ro", "ru", "sk", "sr", "sv", "ta", "te", "uk", "vi", "zh"]  

#     def __init__(self, lang: LANGS):
#         super().__init__(
#             name=f"arc-{lang}",
#             prompt_function=get_arc_prompt(lang, nested_choices=True),
#             suite=("custom",),
#             hf_repo="jon-tow/okapi_arc_challenge",
#             hf_subset=lang,
#             hf_revision="823d5d7bfaf8974a3ab52a825b6cf4903b35dbc4",
#             trust_dataset=True,
#             evaluation_splits=("test",),
#             few_shots_split="train",
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