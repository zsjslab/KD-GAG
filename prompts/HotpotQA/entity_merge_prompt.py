def prompt(text, entity_list_1, entity_list_2):
    entity_merge_prompt = f"""Given a piece of text and two lists of entities extracted from it, merge them into one single list, make sure there is no redundant or duplicate entities in your answer.
Text: {text}
Entity list 1: {entity_list_1}
Entity list 2: {entity_list_2}
Answers: """
    return entity_merge_prompt