from typing import get_args

from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string, build_tasks_per_group, TASKS_ENUM

from lighteval.community_tasks.multilingual.tasks.qa.mkqa import MkqaTask, TaskType

from lighteval.community_tasks.multilingual.tasks.mqa.exams import ExamsTask, subjects_by_lang_code
from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string

from lighteval.community_tasks.multilingual.tasks.qa.mintaka import MintakaTask
from lighteval.community_tasks.multilingual.tasks.nli.lambada import LambadaTask
from lighteval.community_tasks.multilingual.tasks.mqa.mlmm import get_mlmm_tasks
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.belebele import BelebeleTask
from lighteval.community_tasks.multilingual.tasks.nli.pawns import PawnsXTask
from lighteval.community_tasks.multilingual.tasks.nli.xcsr import XCODAHTask, XCSQATask
from lighteval.community_tasks.multilingual.tasks.nli.xnli import XNLI2Task, XNLITask
from lighteval.community_tasks.multilingual.tasks.nli.xwinograd import XWinogradeTask
from lighteval.community_tasks.multilingual.tasks.suites.frenchbench import _GENERATIVE_TASKS as _FRENCH_BENCH_GENERATIVE_TASKS, _MC_TASKS as _FRENCH_BENCH_MC_TASKS
from lighteval.community_tasks.multilingual.tasks.mqa.meta_mmlu import MetaMMLUTask, MMLU_SUBSET

_TASKS_LIST = [get_mlmm_tasks, LambadaTask, XWinogradeTask, XNLITask, PawnsXTask, BelebeleTask, MintakaTask, XCSQATask, XCODAHTask]

_TASKS_DICT = build_tasks_per_group(_TASKS_LIST, 'fr')

_GENERATIVE_TASKS = _TASKS_DICT[TASKS_ENUM.GENERATIVE_TASKS]
_MC_TASKS = _TASKS_DICT[TASKS_ENUM.MC_TASKS]

_GENERATIVE_TASKS = _GENERATIVE_TASKS + [
    *_FRENCH_BENCH_GENERATIVE_TASKS,
    *[MkqaTask(lang="fr", type=type) for type in get_args(TaskType)]
]

_MC_TASKS = _MC_TASKS + [
    *_FRENCH_BENCH_MC_TASKS,
    *[MetaMMLUTask("fr", subset) for subset in get_args(MMLU_SUBSET)],
    *[ExamsTask(lang="fr", subject=subject, show_options=show_options) for subject in subjects_by_lang_code["fr"] for show_options in [True, False]]
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))


early_signals_generative = [
    "fquadv2",
    "mintaka-fr",
]

early_signals_mc = [
    "belebele-fr",
    "arc-fr",
    "french-hellaswag",
    *[MetaMMLUTask("fr", subset) for subset in get_args(MMLU_SUBSET)],
    "pawns-v2-fr",
    "x-codah-fr",
    "x-csqa-fr",
    "xnli-2.0-bool-v2-fr",
]

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="fr", version=version) for version in (1, 2)] +
                            [XNLI2Task(lang="fr", version=version) for version in (1, 2)] +
                            [PawnsXTask(lang="fr", version=version) for version in (1, 2)]
                            ),
    "xnli2": tasks_to_string([XNLI2Task(lang="fr", version=2)]),
    "meta_mmlu": tasks_to_string([MetaMMLUTask("fr", subset) for subset in get_args(MMLU_SUBSET)]),
    "xcodah": tasks_to_string([XCODAHTask(lang="fr")]),
    "exams": tasks_to_string([ExamsTask(lang="fr", subject=subject, show_options=show_options) for subject in subjects_by_lang_code["fr"] for show_options in [True, False]]),
    "mkqa": tasks_to_string([MkqaTask(lang="fr", type=type) for type in get_args(TaskType)]),
    "winograde": tasks_to_string([XWinogradeTask(lang="fr")]),
    "early-signals-generative": tasks_to_string(early_signals_generative),
    "early-signals-mc": tasks_to_string(early_signals_mc),
    "early-signals": tasks_to_string(early_signals_generative + early_signals_mc),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))