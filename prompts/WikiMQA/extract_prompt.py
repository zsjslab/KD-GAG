def prompt(topic, text):
    extract_prompt = f"""Your task is to transform the given text into a semantic graph in the form of a list of triples. The triples must be in the form of [Entity1, Relationship, Entity2]. In your answer, please strictly only include the triples and do not include any explanation or apologies.

Here are some examples:

Example 1:
Topic: Move (1970 film)
Text: Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg.
Triplets: [['Move (1970 film)', 'release year', '1970'], ['Move (1970 film)', 'country', 'America'], ['Move (1970 film)', 'genre', 'comedy'], ['Move (1970 film)', 'star', 'Elliott Gould'], ['Move (1970 film)', 'star', 'Paula Prentiss'], ['Move (1970 film)', 'star', 'Geneviève Waïte'], ['Move (1970 film)', 'director', 'Stuart Rosenberg']]

Example 2:
Topic: Stuart Rosenberg
Text: 'Stuart Rosenberg (August 11, 1927 – March 15, 2007) was an American film and television director whose motion pictures include "Cool Hand Luke" (1967), "Voyage of the Damned" (1976), "The Amityville Horror" (1979), and "The Pope of Greenwich Village" (1984). He was noted for his work with actor Paul Newman.'
Triplets: [['Stuart Rosenberg', 'date of birth', 'August 11, 1927'], ['Stuart Rosenberg', 'date of death', 'March 15, 2007'], ['Stuart Rosenberg', 'country', 'America'], ['Stuart Rosenberg', 'occupation', 'director'], ['Cool Hand Luke', 'director', 'Stuart Rosenberg'], ['Cool Hand Luke', 'release year', '1967'], ['Voyage of the Damned', 'director', 'Stuart Rosenberg'], ['Voyage of the Damned', 'release year', '1976'], ['The Amityville Horror', 'director', 'Stuart Rosenberg'], ['The Amityville Horror', 'release year', '1979'], ['The Pope of Greenwich Village', 'director', 'Stuart Rosenberg'], ['The Pope of Greenwich Village', 'release year', '1984']]

Example 3:
Topic: Vatroslav Mimica
Text: He had his directorial and screenwriting debut in the 1952 Yugoslav film "In the Storm"( Croatian:" U oluji") which starred Veljko Bulajić, Mia Oremović and Antun Nalis. In the 1950s Mimica worked as a director and writer on a number of critically acclaimed animated films and became a prominent member of the Zagreb School of Animated Films( his 1958 animated short film" The LonerSamac") was awarded the Venice Grand Prix), along with authors such as Vlado Kristl and Academy Award- winning Dušan Vukotić.
Triplets: [['In the Storm', 'director', 'Vatroslav Mimica'], ['In the Storm', 'writer', 'Vatroslav Mimica'], ['In the Storm', 'release year', '1952'], ['In the Storm', 'country', 'Yugoslav'], ['In the Storm', 'title in Croatian', 'U oluji'], ['In the Storm', 'star', 'Veljko Bulajić'], ['In the Storm', 'star', 'Mia Oremović'], ['In the Storm', 'star', 'Antun Nalis'], ['Vatroslav Mimica', 'occupation', 'director'], ['Vatroslav Mimica', 'occupation', 'writer'], ['Vatroslav Mimica', 'worked on', 'animated films'], ['Vatroslav Mimica', 'member of', 'Zagreb School of Animated Films'], ['The LonerSamac', 'director', 'Vatroslav Mimica'], ['The LonerSamac', 'year', '1958'], ['The LonerSamac', 'award', 'Venice Grand Prix'], ['Vatroslav Mimica', 'schoolmate', 'Vlado Kristl'], ['Vlado Kristl', 'member of', 'Zagreb School of Animated Films'], ['Vatroslav Mimica', 'schoolmate', 'Dušan Vukotić'], ['Dušan Vukotić', 'member of', 'Zagreb School of Animated Films']]

Now please extract triplets from the following text.
Topic: {topic}
Text: {text}
Triples: """
    return extract_prompt