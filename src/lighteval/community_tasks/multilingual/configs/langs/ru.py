from typing import get_args

from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string, build_tasks_per_group, TASKS_ENUM

from lighteval.community_tasks.multilingual.tasks.qa.custom_squad import SberSquadTask
from lighteval.community_tasks.multilingual.tasks.qa.mkqa import MkqaTask, TaskType
from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string

from lighteval.community_tasks.multilingual.tasks.suites.mera import GENERATIVE_TASKS as _MERA_GENERATIVE_TASKS, MC_TASKS as _MERA_MC_TASKS, RUMMLU_SUBSET, RCBTask, RuMMLUTask
from lighteval.community_tasks.multilingual.tasks.mqa.mlmm import get_mlmm_tasks
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.belebele import BelebeleTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.m3exam import M3ExamTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xquad import XquadTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from lighteval.community_tasks.multilingual.tasks.nli.xcsr import XCODAHTask, XCSQATask
from lighteval.community_tasks.multilingual.tasks.nli.xnli import XNLITask, XNLI2Task
from lighteval.community_tasks.multilingual.tasks.nli.xwinograd import XWinogradeTask
from lighteval.community_tasks.multilingual.tasks.qa.tydiqa import TydiqaTask

_TASKS_LIST = [XquadTask, TydiqaTask, get_mlmm_tasks, XStoryClozeTask, XWinogradeTask, XNLITask, BelebeleTask, XCSQATask, XCODAHTask]

_TASKS_DICT = build_tasks_per_group(_TASKS_LIST, 'ru')

_GENERATIVE_TASKS = _TASKS_DICT[TASKS_ENUM.GENERATIVE_TASKS]
_MC_TASKS = _TASKS_DICT[TASKS_ENUM.MC_TASKS]

_GENERATIVE_TASKS = _GENERATIVE_TASKS + [
    SberSquadTask(),
    *_MERA_GENERATIVE_TASKS,
    *[MkqaTask(lang="ru", type=task_type) for task_type in get_args(TaskType)]
]

_MC_TASKS = _MC_TASKS + [
    *_MERA_MC_TASKS,
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

early_signals_generative = [
    "tydiqa-ru",
    "sber_squad",
    "xquad-ru",
]

early_signals_mc = [
    "arc-ru",
    "belebele-ru",
    "hellaswag-ru",
    "parus",
    *[RuMMLUTask(subset).name for subset in get_args(RUMMLU_SUBSET)],
    "ruopenbookqa",
    "x-codah-ru",
    "x-csqa-ru",
    "xnli-2.0-bool-v2-ru",
    "xstory_cloze-ru",
]

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([RCBTask(version=version) for version in (1, 2)] + [XNLITask(lang="ru", version=version) for version in (1, 2)] + [XNLI2Task(lang="ru", version=version) for version in (1, 2)]),
    "xnli2": tasks_to_string([XNLI2Task(lang="ru", version=2)]),
    "mkqa": tasks_to_string([MkqaTask(lang="ru", type=task_type) for task_type in get_args(TaskType)]),
    "sber_squad": tasks_to_string([SberSquadTask()]),
    "xcodah": tasks_to_string([XCODAHTask("ru")]),
    "winograde": tasks_to_string([XWinogradeTask("ru")]),
    "openbookqa": tasks_to_string(["ruopenbookqa"]),
    "early-signals": tasks_to_string(early_signals_generative + early_signals_mc),
    "early-signals-generative": tasks_to_string(early_signals_generative),
    "early-signals-mc": tasks_to_string(early_signals_mc),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))