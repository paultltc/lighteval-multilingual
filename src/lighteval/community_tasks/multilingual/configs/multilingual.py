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
#from lighteval.community_tasks.multilingual.tasks.utils.translation_literals import LANGS

import importlib

LANGS = ['ar', 'bg', 'bn', 'ca', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'ru', 'sw', 'ta', 'te', 'th', 'tr', 'ur', 'vi', 'zh']

_ALL_TASKS = []
TASKS_GROUPS = {}

langs_tasks = {lang: [] for lang in LANGS}
for task_group in TASKS_ENUM:
    for task in task_group.value:
        tls = []
        for lang in get_args(task.LANGS):
            # TODO: Decide if we keep this
            # Keep only the subset of desired languages
            if not lang in LANGS:
                continue
            # If the task has multiple subsets, instantiate them all
            if hasattr(task, "SUBSETS"):
                tl = [task(lang=lang, subset=s) for s in get_args(task.SUBSETS)]
                langs_tasks[lang].extend(tl)
                tls.extend(tl)
            # Otherwise, instantiate the task
            else:
                tl = task(lang=lang)
                langs_tasks[lang].append(tl)
                tls.append(tl)
        _ALL_TASKS.extend(tls)
        TASKS_GROUPS[task.NAME] = tasks_to_string(tls)

for lang, lang_tasks in langs_tasks.items():
    # Add lang-level group
    TASKS_GROUPS[lang] = tasks_to_string(lang_tasks)

    # Add pair-level group
    for task in lang_tasks:
        TASKS_GROUPS[f"{task.NAME}-{lang}"] = tasks_to_string([task])

# Add the 'all' group
TASKS_GROUPS['all'] = tasks_to_string(_ALL_TASKS)

TASKS_TABLE = [task for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))