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

_TASKS_LIST = [MlqaTask, XquadTask, get_mlmm_tasks, XCopaTask, XNLITask, M3ExamTask, BelebeleTask, XCSQATask, XCODAHTask]

_TASKS_DICT = build_tasks_per_group(_TASKS_LIST, 'vi')

_GENERATIVE_TASKS = _TASKS_DICT[TASKS_ENUM.GENERATIVE_TASKS]
_MC_TASKS = _TASKS_DICT[TASKS_ENUM.MC_TASKS]

_ALL_TASKS = _GENERATIVE_TASKS + _MC_TASKS

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))