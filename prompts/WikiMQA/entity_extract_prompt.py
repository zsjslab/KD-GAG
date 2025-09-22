def prompt(topic, text):
    entity_extract_prompt = f"""Your task is to extract entities from the given text. In your answer, please strictly only include the entities and do not include any explanation or apologies.

Here are some examples:

Example 1:
Topic: Move (1970 film)
Text: Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg.
Entities: ['Move (1970 film)', '1970', 'American', 'comedy', 'Elliott Gould', 'Paula Prentiss', 'Geneviève Waïte', 'Stuart Rosenberg']

Example 2:
Topic: Stuart Rosenberg
Text: Stuart Rosenberg (August 11, 1927 – March 15, 2007) was an American film and television director whose motion pictures include "Cool Hand Luke" (1967), "Voyage of the Damned" (1976), "The Amityville Horror" (1979), and "The Pope of Greenwich Village" (1984). He was noted for his work with actor Paul Newman.
Entities: ['Stuart Rosenberg', 'August 11, 1927', 'March 15, 2007', 'American', 'director', 'Cool Hand Luke', '1967', 'Voyage of the Damned', '1976', 'The Amityville Horror', '1979', 'The Pope of Greenwich Village', '1984', 'Paul Newman']

Example 3:
Topic: Vatroslav Mimica
Text: He had his directorial and screenwriting debut in the 1952 Yugoslav film "In the Storm"( Croatian:" U oluji") which starred Veljko Bulajić, Mia Oremović and Antun Nalis. In the 1950s Mimica worked as a director and writer on a number of critically acclaimed animated films and became a prominent member of the Zagreb School of Animated Films( his 1958 animated short film" The LonerSamac") was awarded the Venice Grand Prix), along with authors such as Vlado Kristl and Academy Award- winning Dušan Vukotić.
Entities: ['Vatroslav Mimica', '1952', 'Yugoslav', 'In the Storm', 'Croatian', 'U oluji', 'Veljko Bulajić', 'Mia Oremović', 'Antun Nalis', '1950s', 'director', 'writer', 'animated films', 'Zagreb School of Animated Films', '1958', 'The Loner', 'Samac', 'Venice Grand Prix', 'Vlado Kristl', 'Dušan Vukotić']


Now please extract entities from the following text.
Topic: {topic}
Text: {text}
Entities: """
    return entity_extract_prompt