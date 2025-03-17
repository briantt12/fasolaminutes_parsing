#!/usr/bin/env python
# encoding: utf-8
import pandas as pd
import re
import util
import spacy
from fuzzywuzzy import process
nlp = spacy.load("en_core_web_sm")

cached_non_name_words = set()  # Cache of confirmed non-names
cached_name_words = set()  # Cache of confirmed names
multi_name_words = set()



# def is_name(word):
#     """Check if a word is a name using caching and spaCy."""
#     if word in cached_name_words:
#         return True
#     if word in cached_non_name_words:
#         return False

#     doc = nlp(word)
#     for token in doc:
#         if token.ent_type_ == "PERSON":
#             cached_name_words.add(word)
#             return True
#         elif token.pos_ == "PROPN":
#             cached_name_words.add(word)
#             return True
#         else:
#             cached_non_name_words.add(word)  # Cache non-names
#     return False



def is_name(word):
    """Check if a word is a name using caching, fuzzy matching, and spaCy."""
    if word in cached_name_words:
        return True
    if word in cached_non_name_words:
        return False

    # names = word.split()
    if "." in word: # Only apply fuzzy matching if word has more than 2 components
        if word not in multi_name_words:
            multi_name_words.add(word) # Cache multi-word names for future lookups

        # Use fuzzy matching only if we have other multi-word names stored
        if multi_name_words:
            match, score = process.extractOne(word, multi_name_words)
            if score >= 85: # Adjust threshold as needed
                return True

    # Perform NLP-based name detection
    doc = nlp(word)
    for token in doc:
        if token.ent_type_ == "PERSON" or token.pos_ == "PROPN":
            cached_name_words.add(word)
            return True

    cached_non_name_words.add(word) # Cache non-names
    return False





def clean_ner(name):
    """Process names and avoid redundant spaCy checks."""
    name = name.strip()
    if is_name(name):
        return name, True  # Return name and confirmation that it's a name
    return name, False

bad_words = [
    'Chairman',
    'Chairperson',
    'Chairwoman',
    'Chairmen',
    'Chair',
    'Chairlady',
    'Chairpersons',
    'Chairs',
    'Co-Chair',
    'Co-Chairs',
    'Co-Chairmen',
    'Co-Chairperson',
    'Co-chair',
    'Co-chairs',
    'Singing',
    'Secretary',
    'Secretaries',
    'Treasurer',
    'Treasurers',
    'Director',
    'Committee',
    'Committees',
    'Convention',
    'Sacred',
    'Musical',
    'Methodist',
    'Baptist',
    'Episcopal',
    'Anglican',
    'Mennonite',
    'Catholic',
    'Pastor',
    'Minister',
    'Ministry',
    'National',
    'Library',
    'Shape',
    'Note',
    'State',
    'Sacra',
    'United',
    'Memorial',
    'Alabama',
    'Mississippi',
    'Arrangement',
    'Arrangements',
    'Arranging',
    'College',
    'University',
    'Courthouse',
    'Meetinghouse',
    'Meeting House'
    'Friends',
    'Seminary',
    'Cemetary',
    'SHMHA',
    'Seek',
    'President',
    'Vice',
    'After',
    'Then',
    'Academy',
    'Officers',
    'Chaplain',
    'Fasola',
    'FaSoLa',
    'Southern',
    'Western',
    'Northwestern',
    'North',
    'Pacific',
    'Northern',
    'International',
    'Center',
    'African',
    'American',
    'School',
    'Elementary',
    'Highway',
    'Outgoing',
    'Recreation',
    'City',
    'County',
    'Avenue',
    'Public',
    'Publishing',
    'Primitive',
    'Mountain',
    'Annual',
    'Department',
    'Presbyterian',
    'Conference',
    'Railroad',
    'Society',
    'Historical',
    'Association',
    'Professor',
    'Associate',
    'Municipal',
    'Building',
    'Labor',
    'County',
    'Line',
    'Elder',
    'Resolutions',
    'Father',
    'Moderator']


non_denson = [
    'ACH',
    'AH',
    'AV',
    'CB',
    'CH',
    'EH\s1',
    'EH\s2',
    'EH1',
    'EH2',
    'SoH',
    'GH',
    'HS',
    'ICH',
    'KsH',
    'KH',
    'LD',
    'MH',
    'NH',
    'NHC',
    'OSH',
    'ShH',
    'ScH',
    'WB']


def build_bad_words():
    ss = ''
    for s in bad_words:
        ss += s + '[\.\s,’]+|'
    ss = ss[:-1]
    return ss


def build_non_denson():
    ss = ''
    for s in non_denson:
        ss += r'\(' + s + r'\)|'
    ss = ss[:-1]
    return ss


def parse_minutes(s, debug_print=False):
    session_count = 0
    sessions = re.split('RECESS|LUNCH',s)
    d = []
    for session in sessions:
        session_count += 1

        # name_pattern = re.compile('(?<=Chairman\s)[A-Z]\.\s[A-Z]\.\s[A-Z]\w+|[A-Z]\.\s[A-Z]\.\s[A-Z]\w+|(?<=Chairman\s)[A-Z][\w]*?\s[A-Z][\w]*?\s[A-Z]\w+|(?<=Chairman\s)[A-Z][\w]*?\s[A-Z]\w+|[A-Z][\w]*?\s[A-Z][\w]*?\s[A-Z]\w+|[A-Z][\w]*?\s[A-Z]\w+');
        name_pattern = re.compile(r'''
            (\A|(?<=\s))
            ((?!''' + build_bad_words() + r''')
            (?<!for\s)
            (
                # Start with upper case...
                [A-Z\u00C0-\u024F] |
                # ...or lower case followed by a string that has upper case
                [a-z](?=[\u00C0-\u024F\w’]*[A-Z\u00C0-\u024F])
            )
            ([\u00C0-\u024F\w’-]+|\.\s|\.)\s?|van\sden\s|Van\sden\s|van\sDen\s){2,5}
        ''', re.UNICODE | re.VERBOSE)
        # pagenum_pattern = re.compile('[\[\{/](\d{2,3}[tb]?)[\]\}]')
        pagenum_pattern = re.compile(r'[\[\{/\s](\d{2,3}[tb]?)([\]\}\s]|$)(?!' + build_non_denson() + r')')

        dd = []
        leaders = re.split(r'\v|called to order|\:\s|(?<=[^\.][^A-Z\]\}])\.(\s|\Z)|(?<=[\]\}”\)])[;\.\:]|;', session)  #double quotes!
        for chunk in leaders:
            if chunk and (len(chunk) > 2):
                if debug_print: print(chunk)
                songs = re.finditer(pagenum_pattern, chunk)
                first_song = None
                for song in songs:
                    if not first_song:
                        first_song = song
                    pagenum = song.group(1)
                    # print pagenum
                    leaders = re.finditer(name_pattern, chunk)
                    for leader in leaders:
                        if leader.end() <= first_song.start()+1:
                            name = leader.group(0)
                            name = name.strip() # TODO: should be able to incorporate this into regex......

                            name = clean_ner(name)[0]

                            dd.append({'name': name, 'song': pagenum})
                            if debug_print: print('***name: ' + name + '\tsong: ' + pagenum)
                        # else:
                            # print "%d %d"%(leader.end(), first_song.start())
                if debug_print: print("---chunk----------")

        d.append({'session': session_count, 'leaders': dd})
        # print "---session----------"
    # print d
    return d

LEADERS = {} # leader -> id
SONGS = {}   # page -> id
ALIASES = {} # alias -> name
INVALID = set()
def insert_minutes(conn, d, minutes_id, debug_print=False):

    curs = conn.cursor()
    # Seed dicts
    if not SONGS:
        for (id, page) in curs.execute("SELECT id, PageNum FROM songs"):
            SONGS[page] = id
        for (name, alias) in curs.execute("SELECT name, alias FROM leader_name_aliases"):
            ALIASES[alias] = ALIASES.get(alias, name) # don't overwrite existing
        for (name,) in curs.execute("SELECT name FROM leader_name_invalid"):
            INVALID.add(name)

    for session in d:
        for leader in session['leaders']:

            #get song_id
            song_id = SONGS.get(leader['song'])
            if not song_id:
                if leader['song'][-1:] == 't' or leader['song'][-1:] == 'b':
                    #check for song without "t" or "b"
                    song_id = SONGS.get(leader['song'][0:-1])
                else:
                    #check for song on "top"
                    song_id = SONGS.get(leader['song']+'t')
                SONGS[leader['song']] = song_id # memoize this result
            if not song_id:
                print(leader)
                print("\tno song id! %s"%(leader['song']))
                continue

            #find leader by name if exists, create if not
            name = leader['name']

            if name in INVALID:
                if debug_print: print("invalid name! %s" % (name))
                continue

            real_name = ALIASES.get(name)
            if real_name:
                if debug_print: print("replacing %s with %s" % (name, real_name))
                name = real_name

            if name == '?':
                # marked as a "bad" name in the alias table so let's just ignore this altogether
                continue

            leader_id = LEADERS.get(name)
            if not leader_id:
                curs.execute("INSERT INTO leaders (name) VALUES (?)", [name])
                leader_id = curs.lastrowid
                curs.execute("UPDATE leader_name_aliases SET leader_id=? WHERE name=?", [leader_id, name])
                LEADERS[name] = leader_id

            if song_id and leader_id and minutes_id:
                curs.execute("INSERT INTO song_leader_joins (song_id, leader_id, minutes_id) VALUES (?,?,?)", (song_id, leader_id, minutes_id))
            else:
                print("problem?! %d %d %d" % (song_id, leader_id, minutes_id))

    curs.close()


def parse_all_minutes(conn):
    curs = conn.cursor()

    # 3928 - camp fasola 2012
    # 3542 - ireland
    curs.execute("SELECT Minutes, Name, Date, id, isDenson, isVirtual FROM minutes")
    rows = curs.fetchall()
    for row in rows:

        if row[4] == 0 or row[5] == 1:
            continue

        print("%s on %s" % (row[1], row[2]))

        s = row[0]
        d = parse_minutes(s)

        minutes_id = row[3]
        insert_minutes(conn, d, minutes_id)

    conn.commit()
    curs.close()


def parse_minutes_by_id(conn, minutes_id):
    curs = conn.cursor()

    # 3928 - camp fasola 2012
    # 3542 - ireland
    curs.execute("SELECT Minutes, Name, Date, id, isDenson FROM minutes WHERE id=?", [minutes_id])
    rows = curs.fetchall()
    for row in rows:

        if row[4] == 0:
            continue

        print("%s on %s"%(row[1],row[2]))

        s = row[0]
        d = parse_minutes(s)

        minutes_id = row[3]
        insert_minutes(conn, d, minutes_id)
        conn.commit()

    curs.close()


def clear_minutes(conn):
    curs = conn.cursor()
    curs.execute("DELETE FROM leaders")
    curs.execute("DELETE FROM song_leader_joins")
    curs.execute("DELETE FROM sqlite_sequence WHERE name='leaders'")
    curs.execute("DELETE FROM sqlite_sequence WHERE name='song_leader_joins'")
    conn.commit()
    curs.close()


if __name__ == '__main__':
    db = util.open_db()
    clear_minutes(db)
    parse_all_minutes(db)
    pd.DataFrame({"Name Words": list(cached_name_words)}).to_csv("spacy_names.csv", index=False)
    print("name count:",len(cached_name_words))
    pd.DataFrame({"Non-Name Words": list(cached_non_name_words)}).to_csv("spacy_non_names.csv", index=False)
    print("non name count:",len(cached_non_name_words))
    shared_values = cached_name_words & cached_non_name_words  # Intersection of sets
    pd.DataFrame({"Unclear Words": list(shared_values)}).to_csv("spacy_shared_unclear_names.csv", index=False)
    pd.DataFrame({"Multi-Name Words": list(multi_name_words)}).to_csv("spacy_multi_names.csv", index=False)
    print("multi name count:",len(multi_name_words))


    # parse_minutes_by_id(db, 5165)
    db.close()
