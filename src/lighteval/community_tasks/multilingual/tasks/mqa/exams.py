from typing import Literal, get_args

from ..utils.prompts import get_m_exams_prompt
from ..utils.translation_literals import LANG_NAMES_INVERTED
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["ar", "bg", "hr", "hu", "it", "mk", "pl", "pt", "sq", "sr", "tr", "vi", "fr"]


subjects_by_lang_code = {
    'ar': Literal['Biology', 'Islamic Studies', 'Physics', 'Science', 'Social'],
    'bg': Literal['Biology', 'Chemistry', 'Geography', 'History', 'Philosophy', 'Physics'],
    'hr': Literal['Biology', 'Chemistry', 'Ethics', 'Fine Arts', 'Geography', 'Geology', 'History', 'Informatics', 'Philosophy', 'Physics', 'Politics', 'Psychology', 'Religion', 'Sociology'],
    'hu': Literal['Agriculture', 'Agriculture (Mechanical knowledge)', 'Biology', 'Chemistry', 'Economics', 'Economics & Marketing', 'Economics Basics (Business)', 'Economics Basics (Theoretical)', 'Forestry', 'Geography', 'Landscaping', 'Physics', 'Politics', 'Tourism'],
    'it': Literal['Biology', 'Chemistry', 'Ethics', 'Geography', 'Geology', 'History', 'Informatics', 'Philosophy', 'Physics', 'Politics', 'Psychology', 'Sociology'],
    'sr': Literal['Biology', 'Chemistry', 'Ethics', 'Geography', 'Geology', 'History', 'Informatics', 'Philosophy', 'Physics', 'Politics', 'Psychology', 'Religion', 'Sociology'],
    'fr': Literal['Economics', 'Economics & Marketing', 'Economics Basics (Theoretical)', 'Geography', 'Physics'],
    'de': Literal['Chemistry', 'Economics', 'Economics & Marketing', 'Economics Basics (Theoretical)', 'Geography', 'Physics', 'Tourism'],
    'es': Literal['Geography', 'Physics'],
    'lt': Literal['Geology', 'History'],
    'sq': Literal['Biology', 'Business', 'Chemistry', 'Fine Arts', 'History', 'Philosophy', 'Physics', 'Sociology'],
    'mk': Literal['Biology', 'Business', 'Chemistry', 'Fine Arts', 'History', 'Philosophy', 'Physics', 'Sociology'],
    'tr': Literal['Biology', 'Business', 'Chemistry', 'Geography', 'History', 'Philosophy', 'Physics', 'Sociology'],
    'pl': Literal['Professional'],
    'pt': Literal['Biology', 'Economics', 'Geology', 'Philosophy'],
    'vi': Literal['Biology', 'Chemistry', 'Citizenship', 'Geography', 'History', 'Physics']
}

# If too hard we can add help with para
class ExamsTask(LightevalTaskConfig):
    NAME = "exams"
    LANGS = Literal["ar", "bg", "hr", "hu", "it", "mk", "pl", "pt", "sq", "sr", "tr", "vi", "fr"]
    SUBSETS = {
    'ar': {'Biology', 'Islamic Studies', 'Physics', 'Science', 'Social'},
    'bg': {'Biology', 'Chemistry', 'Geography', 'History', 'Philosophy', 'Physics'},
    'hr': {'Biology', 'Chemistry', 'Ethics', 'Fine Arts', 'Geography', 'Geology', 'History', 'Informatics', 'Philosophy', 'Physics', 'Politics', 'Psychology', 'Religion', 'Sociology'},
    'hu': {'Agriculture', 'Agriculture (Mechanical knowledge)', 'Biology', 'Chemistry', 'Economics', 'Economics & Marketing', 'Economics Basics (Business)', 'Economics Basics (Theoretical)', 'Forestry', 'Geography', 'Landscaping', 'Physics', 'Politics', 'Tourism'},
    'it': {'Biology', 'Chemistry', 'Ethics', 'Geography', 'Geology', 'History', 'Informatics', 'Philosophy', 'Physics', 'Politics', 'Psychology', 'Sociology'},
    'sr': {'Biology', 'Chemistry', 'Ethics', 'Geography', 'Geology', 'History', 'Informatics', 'Philosophy', 'Physics', 'Politics', 'Psychology', 'Religion', 'Sociology'},
    'fr': {'Economics', 'Economics & Marketing', 'Economics Basics (Theoretical)', 'Geography', 'Physics'},
    'de': {'Chemistry', 'Economics', 'Economics & Marketing', 'Economics Basics (Theoretical)', 'Geography', 'Physics', 'Tourism'},
    'es': {'Geography', 'Physics'},
    'lt': {'Geology', 'History'},
    'sq': {'Biology', 'Business', 'Chemistry', 'Fine Arts', 'History', 'Philosophy', 'Physics', 'Sociology'},
    'mk': {'Biology', 'Business', 'Chemistry', 'Fine Arts', 'History', 'Philosophy', 'Physics', 'Sociology'},
    'tr': {'Biology', 'Business', 'Chemistry', 'Geography', 'History', 'Philosophy', 'Physics', 'Sociology'},
    'pl': {'Professional'},
    'pt': {'Biology', 'Economics', 'Geology', 'Philosophy'},
    'vi': {'Biology', 'Chemistry', 'Citizenship', 'Geography', 'History', 'Physics'}
    }

    def __init__(self, lang: LANGS, subject: str, show_options: bool=False):
        assert subject in subjects_by_lang_code.get(lang, {}), f"Unsuported {lang=} or {subject=}"
        lang_name = LANG_NAMES_INVERTED[lang].capitalize()
        super().__init__(
            name=f"exams{'_options' if show_options else ''}-{lang}:{subject}",
            prompt_function=get_m_exams_prompt(lang, show_options=show_options),
            suite=("custom",),
            hf_repo="mhardalov/exams",
            hf_subset=f"multilingual",
            evaluation_splits=("test",),
            # Weird bug in dataset
            filter=lambda x: x["answerKey"] != "@" and x["info"]["language"] == lang_name and x["info"]["subject"] == subject,
            few_shots_split="train",
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token, Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi),
        )
    
    def get_lang_tasks(self, lang):
        return [ExamsTask(lang, subset) for subset in get_args(self.SUBSETS[lang])]
    
def get_exams_tasks(lang):
        return [ExamsTask(lang, subset) for subset in get_args(ExamsTask.LANG_SUBSETS[lang])]