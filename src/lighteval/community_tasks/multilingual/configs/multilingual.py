from lighteval.community_tasks.multilingual.tasks.utils.tasks_helpers import tasks_to_string, build_tasks_per_group, TASKS_ENUM
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.belebele import BelebeleTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.m3exam import M3ExamTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xquad import XquadTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from lighteval.community_tasks.multilingual.tasks.mqa.mlmm import get_mlmm_tasks
from lighteval.community_tasks.multilingual.tasks.mqa.xcopa import XCopaTask
from lighteval.community_tasks.multilingual.tasks.nli.lambada import LambadaTask
from lighteval.community_tasks.multilingual.tasks.nli.pawns import PawnsXTask
from lighteval.community_tasks.multilingual.tasks.nli.xcsr import XCODAHTask, XCSQATask
from lighteval.community_tasks.multilingual.tasks.nli.xnli import XNLITask, XNLI2Task
from lighteval.community_tasks.multilingual.tasks.nli.xwinograd import XWinogradeTask
from lighteval.community_tasks.multilingual.tasks.qa.mintaka import MintakaTask
from lighteval.community_tasks.multilingual.tasks.qa.mlqa import MlqaTask
from lighteval.community_tasks.multilingual.tasks.qa.tydiqa import TydiqaTask

from typing import Literal, get_args
from lighteval.community_tasks.multilingual.tasks.utils.translation_literals import LANGS

import importlib

SUPPORTED_LANGS = get_args(LANGS)

_ALL_TASKS = []
TASKS_GROUPS = {lang: [] for lang in SUPPORTED_LANGS}

for task_group in TASKS_ENUM:
    for task in task_group.value:
        tls = []
        for lang in get_args(task.LANGS):
            # Keep only the subset of supported languages
            if not lang in SUPPORTED_LANGS:
                continue
            # If the task has multiple subsets, instantiate them all
            if hasattr(task, "SUBSETS"):
                tl = task.get_lang_tasks(lang)
            # Otherwise, instantiate the task
            else:
                tl = [task(lang=lang)]
            # Create the pair-level group
            TASKS_GROUPS[f"{task.NAME}-{lang}"] = tl
            TASKS_GROUPS[lang].extend(tl)
            tls.extend(tl)
        TASKS_GROUPS[task.NAME] = tls
        _ALL_TASKS.extend(tls)

for group, group_tasks in TASKS_GROUPS.items():
    TASKS_GROUPS[group] = tasks_to_string(group_tasks)

# Add the 'all' group
TASKS_GROUPS['all'] = tasks_to_string(_ALL_TASKS)

TASKS_TABLE = [task for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))