def prompt(text, extracted_triples, extracted_relations):
    schema_prompt = f"""You will be given a piece of text and a list of relational triples in the format of [Subject, Relation, Object] extracted from the text. For each relation present in the triples, your task is to write a description to express the meaning of the relation.

Here are some examples:

Example 1:
Text: Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg.
Triplets: [['Move (1970 film)', 'release year', '1970'], ['Move (1970 film)', 'country', 'America'], ['Move (1970 film)', 'genre', 'comedy'], ['Move (1970 film)', 'star', 'Elliott Gould'], ['Move (1970 film)', 'star', 'Paula Prentiss'], ['Move (1970 film)', 'star', 'Geneviève Waïte'], ['Move (1970 film)', 'director', 'Stuart Rosenberg']]
Relations: ['release year', 'country', 'genre', 'star', 'director']
Answer: 
release year: The subject entity was released or made available to the public in the year specified by the object entity.
country: The subject entity originates from or is associated with the country specified by the object entity.
genre: The subject entity belongs to the genre specified by the object entity.
star: The subject entity features the person specified by the object entity as one of its main actors or performers.
director: The subject entity was directed by the person specified by the object entity.


Example 2:
Text: 'Stuart Rosenberg (August 11, 1927 – March 15, 2007) was an American film and television director whose motion pictures include "Cool Hand Luke" (1967), "Voyage of the Damned" (1976), "The Amityville Horror" (1979), and "The Pope of Greenwich Village" (1984). He was noted for his work with actor Paul Newman.'
Triplets: [['Stuart Rosenberg', 'date of birth', 'August 11, 1927'], ['Stuart Rosenberg', 'date of death', 'March 15, 2007'], ['Stuart Rosenberg', 'country', 'America'], ['Stuart Rosenberg', 'occupation', 'director'], ['Cool Hand Luke', 'director', 'Stuart Rosenberg'], ['Cool Hand Luke', 'release year', '1967'], ['Voyage of the Damned', 'director', 'Stuart Rosenberg'], ['Voyage of the Damned', 'release year', '1976'], ['The Amityville Horror', 'director', 'Stuart Rosenberg'], ['The Amityville Horror', 'release year', '1979'], ['The Pope of Greenwich Village', 'director', 'Stuart Rosenberg'], ['The Pope of Greenwich Village', 'release year', '1984']]
Relations: ['date of birth', 'date of death', 'country', 'occupation', 'director', 'release year']
Answer:
date of birth: The subject entity was born on the date specified by the object entity.
date of death: The subject entity died on the date specified by the object entity.
country: The subject entity is from the country specified by the object entity.
occupation: The subject entity has the occupation specified by the object entity.
director: The subject entity directed the film specified by the object entity.
release year: The subject entity was released or made available to the public in the year specified by the object entity.


Example 3:
Text: He had his directorial and screenwriting debut in the 1952 Yugoslav film "In the Storm"( Croatian:" U oluji") which starred Veljko Bulajić, Mia Oremović and Antun Nalis. In the 1950s Mimica worked as a director and writer on a number of critically acclaimed animated films and became a prominent member of the Zagreb School of Animated Films( his 1958 animated short film" The LonerSamac") was awarded the Venice Grand Prix), along with authors such as Vlado Kristl and Academy Award- winning Dušan Vukotić.
Triplets: [['In the Storm', 'director', 'Vatroslav Mimica'], ['In the Storm', 'writer', 'Vatroslav Mimica'], ['In the Storm', 'release year', '1952'], ['In the Storm', 'country', 'Yugoslav'], ['In the Storm', 'title in Croatian', 'U oluji'], ['In the Storm', 'star', 'Veljko Bulajić'], ['In the Storm', 'star', 'Mia Oremović'], ['In the Storm', 'star', 'Antun Nalis'], ['Vatroslav Mimica', 'occupation', 'director'], ['Vatroslav Mimica', 'occupation', 'writer'], ['Vatroslav Mimica', 'worked on', 'animated films'], ['Vatroslav Mimica', 'member of', 'Zagreb School of Animated Films'], ['The LonerSamac', 'director', 'Vatroslav Mimica'], ['The LonerSamac', 'year', '1958'], ['The LonerSamac', 'award', 'Venice Grand Prix'], ['Vatroslav Mimica', 'schoolmate', 'Vlado Kristl'], ['Vlado Kristl', 'member of', 'Zagreb School of Animated Films'], ['Vatroslav Mimica', 'schoolmate', 'Dušan Vukotić'], ['Dušan Vukotić', 'member of', 'Zagreb School of Animated Films']]
Relations: ['director', 'writer', 'release year', 'country', 'star', 'title in Croatian', 'occupation', 'member of', 'schoolmate',]
Answer:
director: The subject entity directed the film specified by the object entity.
writer: The subject entity wrote the film specified by the object entity.
release year: The subject entity was released or made available to the public in the year specified by the object entity.
country: The subject entity originates from or is associated with the country specified by the object entity.
star: The subject entity features the person specified by the object entity as one of its main actors or performers.
title in Croatian: The subject entity has the title specified by the object entity in Croatian.
occupation: The subject entity has the occupation specified by the object entity.
member of: The subject entity is a member of the group or school specified by the object entity.
schoolmate: The subject entity is a colleague or peer of the person specified by the object entity within the same professional context or school.


Now please extract relation descriptions given the following text and triples. Note that the description needs to be general and can be used to describe relations between other entities as well. Pay attention to the order of subject and object entities.
Text: {text}
Triples: {extracted_triples}
Relations: {extracted_relations}
Answer: 
"""
    return schema_prompt