from typing import Literal, get_args

from ..utils.prompts import get_hellaswag_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

# class M_HellaSwagTask(LightevalTaskConfig):
#     NAME = "m_hellaswag"
#     LANGS = Literal["ar", "bn", "ca", "da", "de", "en", "es", "eu", "fr", "gu", "hi", "hr", "hu", "hy", "id", "is", "it", "kn", "ml", "mr", "nb", "ne", "nl", "pt", "ro", "ru", "sk", "sr", "sv", "ta", "te", "uk", "vi", "zh"]

#     def __init__(self, lang: LANGS):
#         super().__init__(
#             name=f"hellaswag-{lang}",
#             prompt_function=get_hellaswag_prompt(lang, use_activity_label=True),
#             suite=("custom",),
#             hf_repo="alexandrainst/m_hellaswag",
#             hf_subset=lang,
#             trust_dataset=True,
#             evaluation_splits=("test",),
#             few_shots_split="val",
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

# ----------------------------------- OKAPI ---------------------------------- #
class M_HellaSwagTask(LightevalTaskConfig):
    NAME = "m_hellaswag"
    LANGS = Literal["ar", "bn", "ca", "da", "de", "es", "eu", "fr", "gu", "hi", "hr", "hu", "hy", "id", "it", "kn", "ml", "mr", "ne", "nl", "pt", "ro", "ru", "sk", "sr", "sv", "ta", "te", "uk", "vi", "zh"]

    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"hellaswag-{lang}",
            prompt_function=get_hellaswag_prompt(lang, use_activity_label=False),
            suite=("custom",),
            hf_repo="jon-tow/okapi_hellaswag",
            hf_subset=lang,
            hf_revision="96ed8e0dfc6172dad1d3df338d7b8ba6c1ff9d83",
            trust_dataset=True,
            evaluation_splits=("validation",),
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