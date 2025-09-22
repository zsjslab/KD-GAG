def prompt(text, query_triplet, query_relation, query_relation_definition, choices):
    sc_prompt = f"""Given the following text and a relational triplet extracted from it:

Text: {text}
Triplet: {query_triplet}

The relation '{query_relation}' in the triplet is defined as '{query_relation_definition}' In this context, is there any relation appropriate to replace it? Please answer by providing only the letter of your choice!

Choices:
{choices}

answer: """
    return sc_prompt