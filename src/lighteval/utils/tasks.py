
def get_lang_from_task(task_name: str):
    # If we pass a full task name extract the task
    if len(task_name.split('|')) > 1:
        return get_lang_from_task(task_name.split('|')[1])
    
    # get lang split
    lang_split = task_name.split('-', maxsplit=1)

    # No langs
    if len(lang_split) == 1:
        return None
    else:
        lang = lang_split[-1].split(':', maxsplit=1)[0].strip()
        return lang