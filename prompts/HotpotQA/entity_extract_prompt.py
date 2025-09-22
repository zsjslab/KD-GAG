def prompt(topic, text):
    entity_extract_prompt = f"""Your task is to extract entities from the given text. In your answer, please strictly only include the entities and do not include any explanation or apologies.

Here are some examples:


Example 1:
Topic: America East Conference
Text: The America East Conference is a collegiate athletic conference affiliated with the NCAA Division I, whose members are located mainly in the Northeastern United States.  The conference was known as the Eastern College Athletic Conference-North from 1979 to 1988 and the North Atlantic Conference from 1988 to 1996.
Entities: ['America East Conference', 'NCAA Division I', 'Northeastern United States', 'Eastern College Athletic Conference-North', '1979 to 1988', 'North Atlantic Conference', '1988 to 1996']

Example 2:
Topic: Old and in the Way (album)
Text: Old and in the Way is the self-titled first album by the bluegrass band Old and in the Way.  It was recorded 8 October 1973 at the Boarding House in San Francisco by Owsley Stanley and Vickie Babcock utilizing eight microphones (four per channel) mixed live onto a stereo Nagra tape recorder.  The caricature album cover was illustrated by Greg Irons.  For many years it was the top selling bluegrass album of all time , until that title was taken by the soundtrack album for O Brother, Where Art Thou.
Entities: ['Old and in the Way', '8 October 1973', 'Boarding House in San Francisco', 'Owsley Stanley', 'Vickie Babcock', 'eight microphones', 'stereo Nagra tape recorder', 'top selling bluegrass album', 'O Brother, Where Art Thou soundtrack']

Example 3:
Topic: Heathcote Williams
Text: John Henley Heathcote-Williams (15 November 1941 – 1 July 2017), known as Heathcote Williams, was an English poet, actor, political activist and dramatist.  He wrote a number of book-length polemical poems including "Autogeddon", "Falling for a Dolphin" and "Whale Nation", which in 1988 became, according to Philip Hoare, "the most powerful argument for the newly instigated worldwide ban on whaling." Williams invented his idiosyncratic "documentary/investigative poetry" style which he put to good purpose bringing a diverse range of environmental and political matters to public attention.
Entities: ['Heathcote Williams', 'John Henley Heathcote-Williams', '15 November 1941', '1 July 2017', 'poet', 'actor', 'political activist', 'dramatist', 'Autogeddon', 'Falling for a Dolphin', 'Whale Nation', '1988', 'documentary investigative poetry']


Now please extract entities from the following text.
Topic: {topic}
Text: {text}
Entities: """
    return entity_extract_prompt