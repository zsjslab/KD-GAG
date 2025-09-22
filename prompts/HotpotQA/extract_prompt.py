def prompt(topic, text):
    extract_prompt = f"""Your task is to transform the given text into a semantic graph in the form of a list of triples. The triples must be in the form of [Entity1, Relationship, Entity2]. In your answer, please strictly only include the triples and do not include any explanation or apologies.

Here are some examples:

Example 1:
Topic: America East Conference
Text: The America East Conference is a collegiate athletic conference affiliated with the NCAA Division I, whose members are located mainly in the Northeastern United States. The conference was known as the Eastern College Athletic Conference-North from 1979 to 1988 and the North Atlantic Conference from 1988 to 1996.
Triplets: [['America East Conference', 'affiliation', 'NCAA Division I'], ['America East Conference', 'main location', 'Northeastern United States'], ['America East Conference', 'former name', 'Eastern College Athletic Conference-North'], ['Eastern College Athletic Conference-North', 'period', '1979 to 1988'], ['America East Conference', 'former name', 'North Atlantic Conference'], ['North Atlantic Conference', 'period', '1988 to 1996']]

Example 2:
Topic: Old and in the Way (album)
Text: Old and in the Way is the self-titled first album by the bluegrass band Old and in the Way. It was recorded 8 October 1973 at the Boarding House in San Francisco by Owsley Stanley and Vickie Babcock utilizing eight microphones (four per channel) mixed live onto a stereo Nagra tape recorder. The caricature album cover was illustrated by Greg Irons. For many years it was the top selling bluegrass album of all time , until that title was taken by the soundtrack album for O Brother, Where Art Thou.
Triplets: [['Old and in the Way', 'album', 'Old and in the Way'], ['Old and in the Way', 'band', 'Old and in the Way'], ['Old and in the Way', 'record date', '8 October 1973'], ['Old and in the Way', 'recording location', 'Boarding House in San Francisco'], ['Old and in the Way', 'producer', 'Owsley Stanley'], ['Old and in the Way', 'producer', 'Vickie Babcock'], ['Old and in the Way', 'recording equipment', 'eight microphones'], ['Old and in the Way', 'recording equipment', 'stereo Nagra tape recorder'], ['Old and in the Way', 'album cover illustrator', 'Greg Irons'], ['Old and in the Way', 'former status', 'top selling bluegrass album'], ['Old and in the Way', 'overtaken by', 'O Brother, Where Art Thou soundtrack']]

Example 3:
Topic: Heathcote Williams
Text: John Henley Heathcote-Williams (15 November 1941 – 1 July 2017), known as Heathcote Williams, was an English poet, actor, political activist and dramatist.  He wrote a number of book-length polemical poems including "Autogeddon", "Falling for a Dolphin" and "Whale Nation", which in 1988 became, according to Philip Hoare, "the most powerful argument for the newly instigated worldwide ban on whaling." Williams invented his idiosyncratic "documentary/investigative poetry" style which he put to good purpose bringing a diverse range of environmental and political matters to public attention.
Triplets: [['Heathcote Williams', 'full name', 'John Henley Heathcote-Williams'], ['Heathcote Williams', 'birth date', '15 November 1941'], ['Heathcote Williams', 'death date', '1 July 2017'], ['John Henley Heathcote-Williams', 'known as', 'Heathcote Williams'], ['Heathcote Williams', 'nationality', 'English'], ['Heathcote Williams', 'occupation', 'poet'], ['Heathcote Williams', 'occupation', 'actor'], ['Heathcote Williams', 'occupation', 'political activist'], ['Heathcote Williams', 'occupation', 'dramatist'], ['Heathcote Williams', 'work', 'Autogeddon'], ['Autogeddon', 'writer', 'Heathcote Williams'], ['Heathcote Williams', 'work', 'Falling for a Dolphin'], ['Falling for a Dolphin', 'writer', 'Heathcote Williams'], ['Heathcote Williams', 'work', 'Whale Nation'], ['Autogeddon', 'writer', 'Heathcote Williams'], ['Whale Nation', 'publication year', '1988'],  ['Heathcote Williams', 'invented Style', 'documentary investigative poetry']]


Now please extract triplets from the following text.
Topic: {topic}
Text: {text}
Triples: """
    return extract_prompt