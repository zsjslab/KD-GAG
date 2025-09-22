def prompt(question):
    topic_entity_prompt = f"""Extract the topic entity from the given question without any additional explanation:
        
Here are some examples:

Example 1: 
Question: Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?
Topic Entity: ['Move', 'Méditerranée']

Example 2:
Question: Who is Rhescuporis I (Odrysian)'s paternal grandfather??
Topic Entity: ['Rhescuporis I (Odrysian)']

Example 3:
Question: Where was the director of film The Fascist born?
Topic Entity: ['The Fascist']

Now please extract the topic entity from the following question. Note that there may be more than one topic entities in the question. 

Question: {question}
Topic Entity:"""
    
    return topic_entity_prompt
    