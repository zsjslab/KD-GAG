def prompt(text, extracted_triples, extracted_relations):
    schema_prompt = f"""You will be given a piece of text and a list of relational triples in the format of [Subject, Relation, Object] extracted from the text. For each relation present in the triples, your task is to write a description to express the meaning of the relation.

Here are some examples:

Example 1:
Text: The America East Conference is a collegiate athletic conference affiliated with the NCAA Division I, whose members are located mainly in the Northeastern United States.  The conference was known as the Eastern College Athletic Conference-North from 1979 to 1988 and the North Atlantic Conference from 1988 to 1996.
Triplets: [['America East Conference', 'affiliation', 'NCAA Division I'], ['America East Conference', 'main location', 'Northeastern United States'], ['America East Conference', 'former name', 'Eastern College Athletic Conference-North'], ['Eastern College Athletic Conference-North', 'period', '1979 to 1988'], ['America East Conference', 'former name', 'North Atlantic Conference'], ['North Atlantic Conference', 'period', '1988 to 1996']]
Relations: ['affiliation', 'main location', 'former name', 'period', 'former name']
Answer: 
affiliation: The subject entity is officially connected to or a member of the organization specified by the object entity.
main location: The primary geographical area where the subject entity's members or activities are located is specified by the object entity.
former name: The subject entity was previously known by the name specified by the object entity.
period: The subject entity operated under the specified name or condition during the time frame indicated by the object entity.
former name: The subject entity was previously known by the name specified by the object entity.


Example 2:
Text: Old and in the Way is the self-titled first album by the bluegrass band Old and in the Way.  It was recorded 8 October 1973 at the Boarding House in San Francisco by Owsley Stanley and Vickie Babcock utilizing eight microphones (four per channel) mixed live onto a stereo Nagra tape recorder.  The caricature album cover was illustrated by Greg Irons.  For many years it was the top selling bluegrass album of all time , until that title was taken by the soundtrack album for O Brother, Where Art Thou.
Triplets: [['Old and in the Way', 'album', 'Old and in the Way'], ['Old and in the Way', 'band', 'Old and in the Way'], ['Old and in the Way', 'recording date', '8 October 1973'], ['Old and in the Way', 'recording location', 'Boarding House in San Francisco'], ['Old and in the Way', 'producer', 'Owsley Stanley'], ['Old and in the Way', 'producer', 'Vickie Babcock'], ['Old and in the Way', 'recording equipment', 'eight microphones'], ['Old and in the Way', 'recording equipment', 'stereo Nagra tape recorder'], ['Old and in the Way', 'album cover illustrator', 'Greg Irons'], ['Old and in the Way', 'former status', 'top selling bluegrass album'], ['Old and in the Way', 'overtaken by', 'O Brother, Where Art Thou soundtrack']]
Relations: ['album', 'band', 'recording date', 'recording location', 'producer', 'recording equipment', 'album cover illustrator', 'former status', 'overtaken by']
Answer:
album: The subject entity is the self-titled album of the band specified by the object entity.
band: The subject entity is an album by the band specified by the object entity.
recording date: The subject entity was recorded on the date specified by the object entity.
recording location: The subject entity was recorded at the location specified by the object entity.
producer: The subject entity was produced by the person or entity specified by the object entity.
recording equipment: The subject entity was recorded using the equipment specified by the object entity.
album cover illustrator: The subject entity's album cover was illustrated by the person specified by the object entity.
former status: The subject entity previously held the status specified by the object entity.
overtaken by: The subject entity's former status was overtaken by the entity specified by the object entity.


Example 3:
Text: John Henley Heathcote-Williams (15 November 1941 – 1 July 2017), known as Heathcote Williams, was an English poet, actor, political activist and dramatist.  He wrote a number of book-length polemical poems including "Autogeddon", "Falling for a Dolphin" and "Whale Nation", which in 1988 became, according to Philip Hoare, "the most powerful argument for the newly instigated worldwide ban on whaling." Williams invented his idiosyncratic "documentary/investigative poetry" style which he put to good purpose bringing a diverse range of environmental and political matters to public attention.
Triplets: [['Heathcote Williams', 'full name', 'John Henley Heathcote-Williams'], ['Heathcote Williams', 'birth date', '15 November 1941'], ['Heathcote Williams', 'death date', '1 July 2017'], ['John Henley Heathcote-Williams', 'known as', 'Heathcote Williams'], ['Heathcote Williams', 'nationality', 'English'], ['Heathcote Williams', 'occupation', 'poet'], ['Heathcote Williams', 'occupation', 'actor'], ['Heathcote Williams', 'occupation', 'political activist'], ['Heathcote Williams', 'occupation', 'dramatist'], ['Heathcote Williams', 'work', 'Autogeddon'], ['Autogeddon', 'writer', 'Heathcote Williams'], ['Heathcote Williams', 'work', 'Falling for a Dolphin'], ['Falling for a Dolphin', 'writer', 'Heathcote Williams'], ['Heathcote Williams', 'work', 'Whale Nation'], ['Autogeddon', 'writer', 'Heathcote Williams'], ['Whale Nation', 'publication year', '1988'],  ['Heathcote Williams', 'invented Style', 'documentary investigative poetry']]
Relations: ['full name', 'birth date', 'death date', 'known as', 'occupation', 'work', 'writer', 'publication year', 'invented Style']
Answer:
full name: The complete name of the subject entity is specified by the object entity.
birth date: The subject entity was born on the date specified by the object entity.
death date: The subject entity died on the date specified by the object entity.
known as: The subject entity is commonly known by the name specified by the object entity.
occupation: The subject entity engaged in the profession or activity specified by the object entity.
work: The subject entity created or is associated with the work specified by the object entity.
writer: The subject entity authored the work specified by the object entity.
publication year: The subject entity was published in the year specified by the object entity.
invented Style: The subject entity created or developed the style specified by the object entity.


Now please extract relation descriptions given the following text and triples. Note that the description needs to be general and can be used to describe relations between other entities as well. Pay attention to the order of subject and object entities.
Text: {text}
Triples: {extracted_triples}
Relations: {extracted_relations}
Answer: 
"""
    return schema_prompt