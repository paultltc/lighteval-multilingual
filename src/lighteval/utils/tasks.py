
def get_lang_from_task(task_name: str):
    # If we pass a full task name extract the task
    if len(task_name.split('|')) > 1:
        return get_lang_from_task(task_name.split('|')[1])
    lang = task_name.split('-', maxsplit=1)[-1].split(':', maxsplit=1)[0].strip()
    return lang
