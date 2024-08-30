from typing import get_args

from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string, build_tasks_per_group, TASKS_ENUM

from lighteval.community_tasks.multilingual.tasks.mqa.custom_hellaswags import CustomHellaswagThaiTask

from lighteval.community_tasks.multilingual.tasks.qa.custom_squad import ThaiQATask

from lighteval.community_tasks.multilingual.tasks.mqa.meta_mmlu import MetaMMLUTask, MMLU_SUBSET

from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string

from lighteval.community_tasks.multilingual.tasks.nli.wsci import WSCITask
from lighteval.community_tasks.multilingual.tasks.mqa.thai_exam import ThaiExamsTask, ThaiExamSubset
from lighteval.community_tasks.multilingual.tasks.mqa.xcopa import XCopaTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.belebele import BelebeleTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.m3exam import M3ExamTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xquad import XquadTask
from lighteval.community_tasks.multilingual.tasks.nli.xnli import XNLITask, XNLI2Task
from lighteval.community_tasks.multilingual.tasks.qa.tydiqa import TydiqaTask
from lighteval.community_tasks.multilingual.tasks.mqa.xcopa import XCopaTask
from lighteval.community_tasks.multilingual.tasks.qa.mkqa import MkqaTask, TaskType

_TASKS_LIST = [TydiqaTask, XquadTask, XCopaTask, XNLITask, M3ExamTask, BelebeleTask]

_TASKS_DICT = build_tasks_per_group(_TASKS_LIST, 'th')

_GENERATIVE_TASKS = _TASKS_DICT[TASKS_ENUM.GENERATIVE_TASKS]
_MC_TASKS = _TASKS_DICT[TASKS_ENUM.MC_TASKS]

_GENERATIVE_TASKS = _GENERATIVE_TASKS + [
    *[MkqaTask(lang="th", type=task_type) for task_type in get_args(TaskType)],
    #ThaiQATask(max_query_length=3200),
]

_MC_TASKS = _MC_TASKS + [
    #CustomHellaswagThaiTask(),
    WSCITask(lang="th"),
    *[ThaiExamsTask(subset=sb) for sb in get_args(ThaiExamSubset)],
    *[MetaMMLUTask("th", subset) for subset in get_args(MMLU_SUBSET)],
]
_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

early_signals_generative = [
    "thaiqa",
    "xquad-th"
]

early_signals_mc = [
    "belebele-th",
    "m3exam-th",
    *[MetaMMLUTask("th", subset) for subset in get_args(MMLU_SUBSET)],
    "xnli-2.0-bool-v2-th",
    "custom_hellaswag-th",
    "thai-exams:tgat"
]

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="th", version=version) for version in (1, 2)] +
                            [XNLI2Task(lang="th", version=version) for version in (1, 2)]),
    "xnli2": tasks_to_string([XNLI2Task(lang="th", version=2)]),
    "meta_mmlu": tasks_to_string([MetaMMLUTask("th", subset) for subset in get_args(MMLU_SUBSET)]),
    "mkqa": tasks_to_string([MkqaTask(lang="th", type=task_type) for task_type in get_args(TaskType)]),
    "wsci": tasks_to_string([WSCITask(lang="th")]),
    "custom_hellaswag": tasks_to_string([CustomHellaswagThaiTask()]),
    "m3exam": tasks_to_string([M3ExamTask(lang="th", version=version) for version in (2,)]),
    "early-signals": tasks_to_string(early_signals_generative + early_signals_mc),
    "early-signals-generative": tasks_to_string(early_signals_generative),
    "early-signals-mc": tasks_to_string(early_signals_mc),
}


TASKS_TABLE = [task.as_dict() for task in _GENERATIVE_TASKS + _MC_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))