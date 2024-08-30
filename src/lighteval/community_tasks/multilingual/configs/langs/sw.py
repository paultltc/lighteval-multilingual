from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string, build_tasks_per_group, TASKS_ENUM

from lighteval.community_tasks.multilingual.tasks.qa.custom_squad import KenswQuADTask
from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string
from lighteval.community_tasks.multilingual.tasks.mqa.xcopa import XCopaTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.m3exam import M3ExamTask
from lighteval.community_tasks.multilingual.tasks.nli.xcsr import XCODAHTask, XCSQATask
from lighteval.community_tasks.multilingual.tasks.nli.xnli import XNLITask, XNLI2Task
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.belebele import BelebeleTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from lighteval.community_tasks.multilingual.tasks.suites.swahili_leaderboard import TASKS as SW_TASKS, ARCSwTask
from lighteval.community_tasks.multilingual.tasks.qa.tydiqa import TydiqaTask

_TASKS_LIST = [XStoryClozeTask, TydiqaTask, XCopaTask, M3ExamTask, BelebeleTask, XCSQATask, XCODAHTask, XNLITask]

_TASKS_DICT = build_tasks_per_group(_TASKS_LIST, 'sw')

_GENERATIVE_TASKS = _TASKS_DICT[TASKS_ENUM.GENERATIVE_TASKS]
_MC_TASKS = _TASKS_DICT[TASKS_ENUM.MC_TASKS]

#_GENERATIVE_TASKS = _GENERATIVE_TASKS + [KenswQuADTask(max_query_length=5000)]

_MC_TASKS = _MC_TASKS + [*SW_TASKS]

_ALL_TASKS = _GENERATIVE_TASKS + _MC_TASKS

early_signals_generative = [
    "kenswquad",
    "tydiqa-sw",
]

early_signals_mc = [
    "belebele-sw",
    "arc-sw:easy",
    "mmlu-sw",
    "m3exam-sw",
    "x-csqa-sw",
    "xcopa-sw",
    "xnli-2.0-bool-v2-sw",
    "xstory_cloze-sw",
]

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="sw", version=version) for version in (1, 2)] +
                            [XNLI2Task(lang="sw", version=version) for version in (1, 2)]),
    "xnli2": tasks_to_string([XNLI2Task(lang="sw", version=2)]),
    "kenswquad": tasks_to_string([KenswQuADTask(max_query_length=6200)]),
    "xcodah": tasks_to_string([XCODAHTask(lang="sw")]),
    "m3exam": tasks_to_string([M3ExamTask(lang="sw", version=version) for version in (2,)]),
    "tydiqa": tasks_to_string([TydiqaTask(lang="sw")]),
    "early-signals": tasks_to_string(early_signals_generative + early_signals_mc),
    "early-signals-generative": tasks_to_string(early_signals_generative),
    "early-signals-mc": tasks_to_string(early_signals_mc),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))