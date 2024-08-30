from typing import get_args

from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string, build_tasks_per_group, TASKS_ENUM

from lighteval.community_tasks.multilingual.tasks.qa.custom_squad import ChineseSQuADTask
from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string
from lighteval.community_tasks.multilingual.tasks.mqa.agieval import CHINESE_AGIEVAL_TASK_TYPE, ChineseAgievalTask, MULTICHOICE_JOIN_VARIANT
from lighteval.community_tasks.multilingual.tasks.mqa.ceval import CEVAL_TASK_TYPE, CEvalTask
from lighteval.community_tasks.multilingual.tasks.mqa.cmmlu import CMMLU_TASK_TYPE, CMMLUTask
from lighteval.community_tasks.multilingual.tasks.mqa.mlmm import M_HellaSwagTask, get_mlmm_tasks
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.belebele import BelebeleTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.m3exam import M3ExamTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xquad import XquadTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from lighteval.community_tasks.multilingual.tasks.nli.clue import CMNLITask, OCNLI, C3Task, CMRC2018Task
from lighteval.community_tasks.multilingual.tasks.nli.pawns import PawnsXTask
from lighteval.community_tasks.multilingual.tasks.nli.xcsr import XCODAHTask, XCSQATask
from lighteval.community_tasks.multilingual.tasks.nli.xnli import XNLITask, XNLI2Task
from lighteval.community_tasks.multilingual.tasks.nli.xwinograd import XWinogradeTask
from lighteval.community_tasks.multilingual.tasks.qa.cmath import CMathTask
from lighteval.community_tasks.multilingual.tasks.qa.mlqa import MlqaTask
from lighteval.community_tasks.multilingual.tasks.qa.tydiqa import TydiqaTask
from lighteval.community_tasks.multilingual.tasks.mqa.xcopa import XCopaTask
from lighteval.community_tasks.multilingual.tasks.qa.mkqa import MkqaTask, TaskType


_TASKS_LIST = [MlqaTask, XquadTask, get_mlmm_tasks, XStoryClozeTask, XWinogradeTask, XCopaTask, XNLI2Task, PawnsXTask, M3ExamTask, BelebeleTask, XCSQATask, XCODAHTask]

_TASKS_DICT = build_tasks_per_group(_TASKS_LIST, 'zh')

_GENERATIVE_TASKS = _TASKS_DICT[TASKS_ENUM.GENERATIVE_TASKS]
_MC_TASKS = _TASKS_DICT[TASKS_ENUM.MC_TASKS]

_GENERATIVE_TASKS = _GENERATIVE_TASKS + [
    #. *[MkqaTask(lang="zh", type=task_type) for task_type in get_args(TaskType)],
    CMathTask(),
    CMRC2018Task(max_generation_chars=100),
    #ChineseSQuADTask(),
]

_MC_TASKS = _MC_TASKS + [
    # CMNLITask(version=2),
    OCNLI(version=2),
    C3Task(),
    *[CMMLUTask(task) for task in get_args(CMMLU_TASK_TYPE)],
    *[CEvalTask(task, show_options=False, join_variant=join_variant) for task in get_args(CEVAL_TASK_TYPE) for join_variant in get_args(MULTICHOICE_JOIN_VARIANT)],
    *[ChineseAgievalTask(task, show_options=False, join_variant=join_variant) for task in get_args(CHINESE_AGIEVAL_TASK_TYPE) for join_variant in get_args(MULTICHOICE_JOIN_VARIANT)],
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

early_signals_generative = [
    "cmrc",
    "mlqa-zh",
    "chinese-squad",
]

early_signals_mc = [
    *[ChineseAgievalTask(task, show_options=False, join_variant="NEW_LINE") for task in get_args(CHINESE_AGIEVAL_TASK_TYPE) if task != "gaokao-mathqa"],
    "belebele-zh",
    "c3",
    *[CEvalTask(task, show_options=False, join_variant="NEW_LINE") for task in get_args(CEVAL_TASK_TYPE)],
    *[CMMLUTask(subset) for subset in get_args(CMMLU_TASK_TYPE)],
    "hellaswag-zh",
    "m3exam-zh",
    "x-codah-zh",
    "x-csqa-zh",
    "xcopa-zh",
    "ocnli-bool-v2-zh",
    "xstory_cloze-zh",
    "xwinograd-zh",
]

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="zh", version=version) for version in (1, 2)] +
                            [XNLI2Task(lang="zh", version=version) for version in (1, 2)] +
                            [OCNLI(version=version) for version in (1, 2)] +
                            [PawnsXTask(lang="zh", version=version) for version in (1, 2)]),
    "xnli2": tasks_to_string([XNLI2Task(lang="zh", version=version) for version in (1, 2)]),
    "ceval": tasks_to_string([CEvalTask(task, show_options=False, join_variant=join_variant) for task in get_args(CEVAL_TASK_TYPE) for join_variant in get_args(MULTICHOICE_JOIN_VARIANT)]),
    "agieval": tasks_to_string([ChineseAgievalTask(task, show_options=False, join_variant=join_variant) for task in get_args(CHINESE_AGIEVAL_TASK_TYPE) for join_variant in get_args(MULTICHOICE_JOIN_VARIANT)]),
    "ocnli": tasks_to_string([OCNLI(version=version) for version in (1, 2)]),
    "mkqa": tasks_to_string([MkqaTask(lang="zh", type=task_type) for task_type in get_args(TaskType)]),
    "xcodah": tasks_to_string([XCODAHTask(lang="zh")]),
    "m3exam": tasks_to_string([M3ExamTask(lang="zh", version=version) for version in (2,)]),
    "cmnli": tasks_to_string([CMNLITask(version=2)]),
    "squad-zh": tasks_to_string([ChineseSQuADTask()]),
    "early-signals-generative": tasks_to_string(early_signals_generative),
    "early-signals-mc": tasks_to_string(early_signals_mc),
    "early-signals": tasks_to_string(early_signals_generative + early_signals_mc),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))