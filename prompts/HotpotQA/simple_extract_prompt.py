def prompt(topic, text):
    extract_prompt = f"""Your task is to transform the given text into a semantic graph in the form of a list of triples. The triples must be in the form of [Entity1, Relationship, Entity2]. In your answer, please strictly only include the triples and do not include any explanation or apologies.
Now please extract triplets from the following text.
Topic: {topic}
Text: {text}
Triples: """
    return extract_prompt