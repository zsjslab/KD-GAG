def prompt(text, relations, entities):
    extract_optim_prompt = f"""Your task is to transform the given text into a semantic graph in the form of a list of triples. The triples must be in the form of [Entity1, Relationship, Entity2]. In your answer, please strictly only include the triples and do not include any explanation or apologies.

Here are some examples:
Example 1:
Text: Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg.
Candidate entities: ['Move (1970 film)', '1970', 'American', 'comedy', 'Elliott Gould', 'Paula Prentiss', 'Geneviève Waïte', 'Stuart Rosenberg']
Triplets: [['Move (1970 film)', 'release year', '1970'], ['Move (1970 film)', 'country', 'America'], ['Move (1970 film)', 'genre', 'comedy'], ['Move (1970 film)', 'star', 'Elliott Gould'], ['Move (1970 film)', 'star', 'Paula Prentiss'], ['Move (1970 film)', 'star', 'Geneviève Waïte'], ['Move (1970 film)', 'director', 'Stuart Rosenberg']]

Example 2:
Text: 'Stuart Rosenberg (August 11, 1927 – March 15, 2007) was an American film and television director whose motion pictures include "Cool Hand Luke" (1967), "Voyage of the Damned" (1976), "The Amityville Horror" (1979), and "The Pope of Greenwich Village" (1984). He was noted for his work with actor Paul Newman.'
Candidate entities: ['Stuart Rosenberg', 'August 11, 1927', 'March 15, 2007', 'American', 'director', 'Cool Hand Luke', '1967', 'Voyage of the Damned', '1976', 'The Amityville Horror', '1979', 'The Pope of Greenwich Village', '1984', 'Paul Newman']
Triplets: [['The Hork-Bajir Chronicles', 'bookType', 'companion'], ['The Hork-Bajir Chronicles', 'series', 'Animorphs'], ['The Hork-Bajir Chronicles', 'author', 'K. A. Applegate'], ['The Hork-Bajir Chronicles', 'numberInSeries', '2']]

Example 3:
Text: He had his directorial and screenwriting debut in the 1952 Yugoslav film "In the Storm"( Croatian:" U oluji") which starred Veljko Bulajić, Mia Oremović and Antun Nalis. In the 1950s Mimica worked as a director and writer on a number of critically acclaimed animated films and became a prominent member of the Zagreb School of Animated Films( his 1958 animated short film" The LonerSamac") was awarded the Venice Grand Prix), along with authors such as Vlado Kristl and Academy Award- winning Dušan Vukotić.
Candidate entities: ['Vatroslav Mimica', '1952', 'Yugoslav', 'In the Storm', 'Croatian', 'U oluji', 'Veljko Bulajić', 'Mia Oremović', 'Antun Nalis', '1950s', 'director', 'writer', 'animated films', 'Zagreb School of Animated Films', '1958', 'The Loner', 'Samac', 'Venice Grand Prix', 'Vlado Kristl', 'Dušan Vukotić']
Triplets: [['In the Storm', 'director', 'Vatroslav Mimica'], ['In the Storm', 'writer', 'Vatroslav Mimica'], ['In the Storm', 'release year', '1952'], ['In the Storm', 'country', 'Yugoslav'], ['In the Storm', 'title in Croatian', 'U oluji'], ['In the Storm', 'star', 'Veljko Bulajić'], ['In the Storm', 'star', 'Mia Oremović'], ['In the Storm', 'star', 'Antun Nalis'], ['Vatroslav Mimica', 'occupation', 'director'], ['Vatroslav Mimica', 'occupation', 'writer'], ['Vatroslav Mimica', 'worked on', 'animated films'], ['Vatroslav Mimica', 'member of', 'Zagreb School of Animated Films'], ['The LonerSamac', 'director', 'Vatroslav Mimica'], ['The LonerSamac', 'year', '1958'], ['The LonerSamac', 'award', 'Venice Grand Prix'], ['Vatroslav Mimica', 'schoolmate', 'Vlado Kristl'], ['Vlado Kristl', 'member of', 'Zagreb School of Animated Films'], ['Vatroslav Mimica', 'schoolmate', 'Dušan Vukotić'], ['Dušan Vukotić', 'member of', 'Zagreb School of Animated Films']]

Now please extract triplets from the following text. Here are some potential relations and their descriptions you may look out for during extraction:
{relations}
Note that this list may not be exhaustive, you may use other relations and not necessarily all relations in this list are present in the text.
Text: {text}
Candidate entities: {entities}"""
    return extract_optim_prompt