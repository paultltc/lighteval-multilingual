from typing import Literal

from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_mlqa_prompt
from lighteval.tasks.lighteval_task import LightevalTaskConfig


# TODO Support all available
#LANGS = Literal["zh", "en", "ar", "hi", "de"]
LANGS = Literal["zh", "en", "ar", "hi", "de", "vi", "es"]


class MlqaTask(LightevalTaskConfig):
    NAME = "mlqa"
    LANGS = Literal["zh", "en", "ar", "hi", "de", "vi", "es"]

    def __init__(self, lang: LANGS, max_query_length: int | None = None):
        self.source_lang = lang
        super().__init__(
            name=f"mlqa-{lang}",
            prompt_function=get_mlqa_prompt(lang),
            suite=("custom",),
            hf_repo="facebook/mlqa",
            hf_subset=f"mlqa.{lang}.{lang}",
            hf_revision="397ed406c1a7902140303e7faf60fff35b58d285",
            filter=lambda x: len(x["question"] + x["context"]) < max_query_length if max_query_length else True,
            trust_dataset=True,
            evaluation_splits=("test",),
            #few_shots_split="train",
            generation_size=100,
            stop_sequence=("\n",),
            metric=(get_qa_metric(lang, "exact"), get_qa_metric(lang, "f1")),
        )
