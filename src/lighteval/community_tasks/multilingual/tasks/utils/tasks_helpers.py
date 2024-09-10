from typing import get_args

from lighteval.community_tasks.multilingual.tasks.mqa_with_context.belebele import BelebeleTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.m3exam import M3ExamTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xquad import XquadTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from lighteval.community_tasks.multilingual.tasks.mqa.exams import ExamsTask
from lighteval.community_tasks.multilingual.tasks.mqa.xcopa import XCopaTask
from lighteval.community_tasks.multilingual.tasks.mqa.m_mmlu import M_MMLUTask
from lighteval.community_tasks.multilingual.tasks.mqa.persian_mmlu import PersianMMLU
from lighteval.community_tasks.multilingual.tasks.mqa.arabic_mmlu import ArabicMMLUTask
from lighteval.community_tasks.multilingual.tasks.mqa.cmmlu import CMMLUTask
from lighteval.community_tasks.multilingual.tasks.nli.lambada import LambadaTask
from lighteval.community_tasks.multilingual.tasks.nli.pawns import PawnsXTask
from lighteval.community_tasks.multilingual.tasks.nli.xcsr import XCODAHTask, XCSQATask
from lighteval.community_tasks.multilingual.tasks.nli.xnli import XNLITask, XNLI2Task
from lighteval.community_tasks.multilingual.tasks.nli.xwinograd import XWinogradeTask
from lighteval.community_tasks.multilingual.tasks.nli.m_hellaswag import M_HellaSwagTask
from lighteval.community_tasks.multilingual.tasks.qa.mintaka import MintakaTask
from lighteval.community_tasks.multilingual.tasks.qa.mlqa import MlqaTask
from lighteval.community_tasks.multilingual.tasks.qa.tydiqa import TydiqaTask
from lighteval.community_tasks.multilingual.tasks.qa.m_truthfulqa import M_TruthfulQATask

from enum import Enum

class TASKS_ENUM(Enum):
    GENERATIVE_TASKS = {
        MintakaTask,
        MlqaTask,
        TydiqaTask,
        XquadTask,
    }

    MC_TASKS = {    
        BelebeleTask,
        LambadaTask,
        PawnsXTask,
        XCODAHTask,
        XCSQATask,
        XNLITask,
        XStoryClozeTask,
        XWinogradeTask,
        M3ExamTask,
        ExamsTask,
        M_HellaSwagTask,
        M_MMLUTask,
        PersianMMLU,
        ArabicMMLUTask,
        CMMLUTask,
        M_TruthfulQATask,
        XCopaTask
    }

def tasks_to_string(tasks: list, n_fewshot: int = 5) -> str:
    return ",".join([f"custom|{t if isinstance(t, str) else t.name}|{n_fewshot if t.few_shots_split else 0}|1" for t in tasks])

def task_to_groups(task) -> TASKS_ENUM:
    groups = []
    for group in TASKS_ENUM:
        if type(task) in group.value:
            groups.append(group)
    return groups

def build_tasks(tasks: list, lang: str) -> list:
    res = []
    for task in tasks:
        build_task = task(lang=lang)
        if isinstance(build_task, list):
            res.extend(build_task)
        else:
            res.append(build_task)
    return res

def build_tasks_per_group(tasks: list, lang: str) -> dict:
    res = {g: [] for g in TASKS_ENUM}  # Initialize with default lists
    proc_tasks = build_tasks(tasks, lang)
    for task in proc_tasks:
        groups = task_to_groups(task)
        for g in groups:
            res[g].append(task)  # Append directly to the group
    return res

def get_available_languages(task):
    return get_args(task.LANGS)
