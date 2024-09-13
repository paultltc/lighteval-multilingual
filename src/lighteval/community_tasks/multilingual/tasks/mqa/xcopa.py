from typing import Literal

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

from ..utils.prompts import get_copa_prompt


LANGS = Literal["ar", "et", "ht", "it", "id", "qu", "sw", "zh", "ta", "th", "tr", "vi"]


class XCopaTask(LightevalTaskConfig):
    NAME = "xcopa"
    LANGS = Literal["ar", "en", "et", "ht", "it", "id", "qu", "sw", "zh", "ta", "th", "tr", "vi"]

    def __init__(self, lang: LANGS):
        repo = "OALL/AlGhafa-Arabic-LLM-Benchmark-Translated" if lang == 'ar' \
                else "xcopa"
        subset = "copa_ext_ar" if lang == "ar" else "translation-zh" if lang == "en" else lang 
        #TODO: The ar also has fewshots
        super().__init__(
            name=f"xcopa-{lang}",
            suite=("custom",),
            prompt_function=get_copa_prompt(lang),
            hf_repo=repo,
            hf_subset=subset,
            evaluation_splits=("test",),
            few_shots_split="validation",
            few_shots_select=None,
            generation_size=-1,
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token, Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi),
            stop_sequence=("\n",),
            trust_dataset=True,
            version=0,
        )
