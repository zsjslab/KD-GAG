def prompt(question):
    topic_entity_prompt = f"""Extract the topic entity from the given question without any additional explanation:
        
Here are some examples:

Example 1: 
Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
Topic Entity: ['Kiss and Tell', 'Corliss Archer']

Example 2:
Question: What is the title of the 1979 film adaptation of William Shakespeare's play in which Heathcote Williams played a main character?
Topic Entity: ['William Shakespeare', '1979', 'Heathcote Williams']

Example 3:
Question: Have filmmakers Enrico Cocozza and Mira Nair both been nominated for awards for their work?
Topic Entity: ['Enrico Cocozza', 'Mira Nair']

Example 4:
Question: What distinction is held by the former NBA player who was a member of the Charlotte Hornets during their 1992-93 season and was head coach for the WNBA team Charlotte Sting?
Topic Entity: ['NBA player', 'Charlotte Hornets', 'head coach', 'Charlotte Sting']


Now please extract the topic entity from the following question. Note that there may be more than one but no more than three topic entities in the question. 

Question: {question}
Topic Entity:"""
    
    return topic_entity_prompt
    