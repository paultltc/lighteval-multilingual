from functools import partial
from typing import Literal, get_args

from ..utils.prompts import get_m_mmlu_prompt, get_mmlu_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

MMLU_SUBSET = Literal[
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

class M_MMLUTask(LightevalTaskConfig):
    NAME = "m_mmlu"
    LANGS = Literal["ar", "bn", "ca", "da", "de", "en", "es", "eu", "fr", "gu", "hi", "hr", "hu", "hy", "id", "is", "it", "kn", "ml", "mr", "nb", "ne", "nl", "pt", "ro", "ru", "sk", "sr", "sv", "ta", "te", "uk", "vi", "zh"]
    SUBSETS = MMLU_SUBSET

    def __init__(self, lang: LANGS, subset: SUBSETS):
        self.subset = subset
        super().__init__(
            name=f"m_mmlu-{lang}:{subset}",
            prompt_function=get_m_mmlu_prompt(lang),
            suite=("custom",),
            hf_repo="alexandrainst/m_mmlu",
            hf_subset=lang,
            filter=lambda line: line["id"].split("/")[0] == subset,
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

    @staticmethod
    def get_lang_tasks(lang):
        return [M_MMLUTask(lang, subset) for subset in get_args(M_MMLUTask.SUBSETS)]

# ----------------------------------- OKAPI ---------------------------------- #
# class M_MMLUTask(LightevalTaskConfig):
#     NAME = "m_mmlu"
#     LANGS = Literal["ar", "bn", "ca", "da", "de", "es", "eu", "fr", "gu", "hi", "hr", "hu", "hy", "id", "it", "kn", "ml", "mr", "ne", "nl", "pt", "ro", "ru", "sk", "sr", "sv", "ta", "te", "uk", "vi", "zh"]
#     SUBSETS = MMLU_SUBSET

#     def __init__(self, lang: LANGS, subset: SUBSETS):
#         super().__init__(
#             name=f"mmlu-{lang}:{subset}",
#             prompt_function=get_mmlu_prompt(lang),
#             suite=("custom",),
#             hf_repo="jon-tow/okapi_mmlu",
#             hf_subset=lang,
#             hf_revision="refs/pr/1",
#             filter=lambda line: line["id"].split("/")[0] == subset,
#             trust_dataset=True,
#             evaluation_splits=("test",),
#             few_shots_split="dev",
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
#         self.subset = subset


def get_m_mmlu_tasks(lang):
    return [M_MMLUTask(lang, subset) for subset in get_args(MMLU_SUBSET)]