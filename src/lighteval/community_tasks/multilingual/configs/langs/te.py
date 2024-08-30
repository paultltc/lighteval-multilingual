from typing import get_args

from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string, build_tasks_per_group, TASKS_ENUM

from lighteval.community_tasks.multilingual.tasks.mqa.custom_hellaswags import CustomHellaswagTeluguTask
from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string
from lighteval.community_tasks.multilingual.tasks.qa.Indicqa import IndicQATask
from lighteval.community_tasks.multilingual.tasks.nli.indicnxnli import XNLIIndicTask
from lighteval.community_tasks.multilingual.tasks.mqa.indicxcopa import XCopaIndicTask
from lighteval.community_tasks.multilingual.tasks.mqa.mlmm import MMLU_SUBSET, get_mlmm_tasks, M_MMLUTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.belebele import BelebeleTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from lighteval.community_tasks.multilingual.tasks.qa.tydiqa import TydiqaTask

_TASKS_LIST = [TydiqaTask, get_mlmm_tasks, BelebeleTask, XStoryClozeTask]

_TASKS_DICT = build_tasks_per_group(_TASKS_LIST, 'te')

_GENERATIVE_TASKS = _TASKS_DICT[TASKS_ENUM.GENERATIVE_TASKS]
_MC_TASKS = _TASKS_DICT[TASKS_ENUM.MC_TASKS]

_GENERATIVE_TASKS = _GENERATIVE_TASKS + [
    IndicQATask(lang="te", max_query_length=2700),
]

_MC_TASKS = _MC_TASKS + [
    XNLIIndicTask(lang="te", version=1),
    XNLIIndicTask(lang="te", version=2),
    XCopaIndicTask(lang="te"),
    CustomHellaswagTeluguTask(),
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

early_signals_generative = [
    "indicqa.te",
]

early_signals_mc = [
    "belebele-te",
    "custom_hellaswag-te",
    *[M_MMLUTask("te", subset) for subset in get_args(MMLU_SUBSET)],
    "indicnxnli-te-bool-v2-te",
    "xcopa-te",
    "xstory_cloze-te",
]

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLIIndicTask(lang="te", version=2)]),
    "custom_hellaswag": tasks_to_string([CustomHellaswagTeluguTask()]),
    "early-signals": tasks_to_string(early_signals_mc + early_signals_generative),
    "early-signals-mc": tasks_to_string(early_signals_mc),
    "early-signals-generative": tasks_to_string(early_signals_generative),
}

TASKS_TABLE = [task.as_dict() for task in _GENERATIVE_TASKS + _MC_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
