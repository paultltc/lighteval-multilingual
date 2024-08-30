from typing import get_args

from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string, build_tasks_per_group, TASKS_ENUM

from lighteval.community_tasks.multilingual.tasks.qa.custom_squad import ChAITask
from lighteval.community_tasks.multilingual.tasks.mqa.meta_mmlu import MetaMMLUTask
from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string
from lighteval.community_tasks.multilingual.tasks.suites.indic_evals import ARCIndTask, BoolQIndTask, HellaSwagIndTask
from lighteval.community_tasks.multilingual.tasks.qa.Indicqa import IndicQATask
from lighteval.community_tasks.multilingual.tasks.mqa.indicxcopa import XCopaIndicTask
from lighteval.community_tasks.multilingual.tasks.nli.indicnxnli import XNLIIndicTask
from lighteval.community_tasks.multilingual.tasks.qa.mintaka import MintakaTask
from lighteval.community_tasks.multilingual.tasks.mqa.mlmm import MMLU_SUBSET, get_mlmm_tasks
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.belebele import BelebeleTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xquad import XquadTask
from lighteval.community_tasks.multilingual.tasks.nli.indicnxnli import XNLIIndicTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from lighteval.community_tasks.multilingual.tasks.nli.xcsr import XCODAHTask, XCSQATask
from lighteval.community_tasks.multilingual.tasks.nli.xnli import XNLITask, XNLI2Task
from lighteval.community_tasks.multilingual.tasks.qa.mlqa import MlqaTask
from lighteval.community_tasks.multilingual.tasks.qa.tydiqa import TydiqaTask

_TASKS_LIST = [MlqaTask, XquadTask, get_mlmm_tasks, XStoryClozeTask, XNLITask, BelebeleTask, MintakaTask, XCSQATask, XCODAHTask]

_TASKS_DICT = build_tasks_per_group(_TASKS_LIST, 'hi')

_GENERATIVE_TASKS = _TASKS_DICT[TASKS_ENUM.GENERATIVE_TASKS]
_MC_TASKS = _TASKS_DICT[TASKS_ENUM.MC_TASKS]

# _GENERATIVE_TASKS = _GENERATIVE_TASKS + [
#     IndicQATask(lang="hi", max_query_length=4300),
#     ChAITask(lang="hi", max_query_length=4300),
#     # BoolQIndTask(),
# ]

# _MC_TASKS = _MC_TASKS + [
#     XNLIIndicTask(lang="hi", version=1),
#     XNLIIndicTask(lang="hi", version=2),
#     XCopaIndicTask(lang="hi"),
#     ARCIndTask(subset="easy"),
#     ARCIndTask(subset="challenge"),
#     HellaSwagIndTask(),
#     BoolQIndTask(),
#     *[MetaMMLUTask("hi", subset) for subset in get_args(MMLU_SUBSET)],
# ]


early_signals_generative = [
    "indicqa.hi",
]
early_signals_mc = [
    "belebele-hi",
    "hellaswag-hi",
    "hi-arc:easy",
    *[MetaMMLUTask("hi", subset) for subset in get_args(MMLU_SUBSET)],
    "x-codah-hi",
    "x-csqa-hi",
    "xcopa-hi",
    "indicnxnli-hi-bool-v2-hi",
    "xstory_cloze-hi",
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="hi", version=version) for version in (1, 2)] +
                            [XNLI2Task(lang="hi", version=version) for version in (1, 2)] +
                            [XNLIIndicTask(lang="hi", version=version) for version in (1, 2)]),
    "xnli2": tasks_to_string([XNLI2Task(lang="hi", version=2)]),
    "meta_mmlu": tasks_to_string([MetaMMLUTask("hi", subset) for subset in get_args(MMLU_SUBSET)]),
    "xcodah": tasks_to_string([XCODAHTask("hi")]),
    "early-signals": tasks_to_string(early_signals_generative + early_signals_mc),
    "early-signals-generative": tasks_to_string(early_signals_generative),
    "early-signals-mc": tasks_to_string(early_signals_mc),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))