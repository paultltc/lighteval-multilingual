from typing import Literal, get_args

from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_mkqa_prompt
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["ar",
            "da",
            "de",
            "en",
            "es",
            "fi",
            "fr",
            "he",
            "hu",
            "it",
            "ja",
            "ko",
            "km",
            "ms",
            "nl",
            "no",
            "pl",
            "pt",
            "ru",
            "sv",
            "th",
            "tr",
            "vi",
            "zh",
        ]

MKQA_SUBSETS = Literal[
    "entity",
    "long_answer",
#    "unanswerable",
    "date",
    "number",
    "number_with_unit",
    "short_phrase",
    "binary"
]


class MkqaTask(LightevalTaskConfig):
    NAME = "mkqa"
    LANGS = Literal["ar","da","de","en","es","fi","fr","he","hu","it","ja","ko","km","ms","nl","no","pl","pt","ru","sv","th","tr","vi","zh"]
    SUBSETS = MKQA_SUBSETS

    def __init__(self, lang: LANGS, type: SUBSETS):
        dst_lang = "zh_cn" if lang == "zh" else lang
        super().__init__(
            name=f"mkqa:{type}-{lang}",
            prompt_function=get_mkqa_prompt(lang, dst_lang),
            suite=("custom",),
            hf_repo="apple/mkqa",
            hf_subset=f"mkqa",
            hf_revision="325131889721ae0ed885b76ecb8011369d75abad",
            filter=lambda line: line["answers"][dst_lang][0]["type"] == get_args(self.SUBSETS).index(type),
            trust_dataset=True,
            evaluation_splits=("train",),
            generation_size=50,
            stop_sequence=("\n",),
            metric=(get_qa_metric(lang, "exact"), get_qa_metric(lang, "f1")),
        )
