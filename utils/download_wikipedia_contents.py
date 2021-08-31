# downloads wikipedia contents of the names given in all_names list and stores all into a txt file
import wikipedia as wk

all_names = ['Barack Obama', 'Donald Trump', 'Hillary Clinton', 'George Bush', 'Bill Clinton', 'Bernie Sanders',
             'Joseph Robinette Biden', 'Kamala Devi Harris', 'Nancy Pelosi', 'Michael Richard Pence']

wiki_contents = ''
for names in all_names:
    wiki_contents += str(wk.page(names).content)

with open('../data/wikipedia_context_files.txt', 'w+', encoding='UTF-8') as f:
    f.write(wiki_contents)
