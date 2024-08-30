from typing import get_args

from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string, build_tasks_per_group, TASKS_ENUM

from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string
from lighteval.community_tasks.multilingual.tasks.mqa.exams import ExamsTask, subjects_by_lang_code
from lighteval.community_tasks.multilingual.tasks.qa.tquad import Tquad2Task
from lighteval.community_tasks.multilingual.tasks.suites.turkish_leaderboard import ARCEasyTrTask, HellaSwagTrTask, MMLUTaskTr, TruthfulQATrTask, WinogradeTrTask, MMLU_SUBSETS
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.belebele import BelebeleTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xquad import XquadTask
from lighteval.community_tasks.multilingual.tasks.nli.xnli import XNLITask, XNLI2Task
from lighteval.community_tasks.multilingual.tasks.mqa.xcopa import XCopaTask
from lighteval.community_tasks.multilingual.tasks.qa.mkqa import MkqaTask, TaskType

_TASKS_LIST = [XquadTask, XCopaTask, XNLITask, BelebeleTask]

_TASKS_DICT = build_tasks_per_group(_TASKS_LIST, 'tr')

_GENERATIVE_TASKS = _TASKS_DICT[TASKS_ENUM.GENERATIVE_TASKS]
_MC_TASKS = _TASKS_DICT[TASKS_ENUM.MC_TASKS]

_GENERATIVE_TASKS = _GENERATIVE_TASKS + [
    Tquad2Task(max_query_length=2800),
    *[MkqaTask(lang="tr", type=task_type) for task_type in get_args(TaskType)]
]

_MC_TASKS = _MC_TASKS + [
    HellaSwagTrTask(),
    TruthfulQATrTask("mc1"),
    TruthfulQATrTask("mc2"),
    ARCEasyTrTask(version=2),
    WinogradeTrTask(),
    *[MMLUTaskTr(subset) for subset in get_args(MMLU_SUBSETS)],
    *[ExamsTask(lang="tr", subject=subject, show_options=show_options) for subject in subjects_by_lang_code["tr"] for show_options in [True, False]]
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))


early_signals_generative = [
    "xquad-tr",
    "tqduad2",
]

early_signals_mc = [
    "arc-v2-tr",
    "belbele-tr",
    *[ExamsTask(lang="tr", subject=subject, show_options=False) for subject in get_args(subjects_by_lang_code["tr"])],
    "hellaswag-tr",
    *[MMLUTaskTr(subset) for subset in get_args(MMLU_SUBSETS)],
    "xcopa-tr",
    "xnli-2.0-bool-v2-tr",
]
    
    
TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="tr", version=version) for version in (1, 2)] +
                            [XNLI2Task(lang="tr", version=version) for version in (1, 2)]),
    "xnli2": tasks_to_string([XNLI2Task(lang="tr", version=2)]),
    "arc": tasks_to_string([ARCEasyTrTask(version=2)]),
    "exams": tasks_to_string([ExamsTask(lang="tr", subject=subject, show_options=show_options) for subject in subjects_by_lang_code["tr"] for show_options in [True, False]]),
    "mkqa": tasks_to_string([MkqaTask(lang="tr", type=task_type) for task_type in get_args(TaskType)]),
    "hellaswag": tasks_to_string([HellaSwagTrTask()]),
    "winograde": tasks_to_string([WinogradeTrTask()]),
    "early-signals": tasks_to_string(early_signals_mc + early_signals_generative),
    "early-signals-mc": tasks_to_string(early_signals_mc),
    "early-signals-generative": tasks_to_string(early_signals_generative),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))