#!/usr/bin/env python
# encoding: utf-8
#original
import pandas as pd
import re
import util
import spacy
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt
import time
nlp = spacy.load("en_core_web_sm")
batch = 1000
confidence = 87
names_count = []
# Caching sets
cached_non_name_words = set()
# passed_names = set()  # Store full names for fuzzy matching
matched = set()
final_words = set()
failed = set()
fuzzyCount = 0
start_time = time.time()
history = set()
original_names = set()

def convert_to_initials(name):
    """Convert the second name to an initial if there are more than two words."""
    original_names.add(name)
    words = name.split()
    if '-' in name:
        firstname = words[0]
        lastname = name.split('-')[-1]
        return firstname+' '+lastname
    
    if len(words) >= 3:  # If there's a middle name or extra names
        return f"{words[0]} {words[1][0]}. {words[-1]}"  # Convert only the second word to an initial
    if len(words) == 2:
        return name  # Keep first and last name as is
    return name  # Return as is if single word

def ner_check_person_is_name(text):
    """Check if a word is a person's name using spaCy NER and POS tagging."""
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return True, ent.text  # Detected as a named entity (PERSON)

    for token in doc:
        if token.pos_ == "PROPN":
            return True, token.text  # Detected as a proper noun

    return False, "fail:" + text  # Neither detected


def is_name(word,confidence):
    global start_time, fuzzyCount

    if(len(final_words)% batch == 0):
        
        pd.DataFrame({"Passed Words Sorted": sorted(final_words)}).to_csv("passed_names_" + str(batch) + ".csv", index=False)
        
        pd.DataFrame({"Matched Words": sorted(list(matched))}).to_csv("matched_names.csv", index=False)
        names_count.append(len(final_words))
        
        
    converted_name = convert_to_initials(word)
    
    
    
    if converted_name in final_words:
        return True, converted_name

    elif converted_name in cached_non_name_words:
        return False, "fail:" + word  # Cached as non-name

    
    if len(converted_name.split()) >= 3:
        firstname = converted_name.split()[0]
        lastname = converted_name.split()[-1]
        shortened_name = f"{firstname} {lastname}"
        
        # Try exact match first
        if shortened_name in final_words:
            return True, shortened_name
        
        # Fuzzy match on shortened name
        ratio_match = process.extractOne(shortened_name, final_words, scorer=fuzz.ratio)
        if ratio_match:
            suggested_word, score, _ = ratio_match
            matched_check = score >= confidence
            history.add((word, suggested_word, score, matched_check)) 
            
            if (suggested_word[0] == shortened_name[0]) and (score >= confidence):
                fuzzyCount += 1
                matched.add(f"{shortened_name}:{suggested_word}: ratio:{score}")
                return True, suggested_word
    # Only fall back for less than 3 names
    elif len(converted_name.split()) < 3:
        ratio_match = process.extractOne(converted_name, final_words, scorer=fuzz.ratio)
        
        if ratio_match:
            suggested_word, score, _ = ratio_match
            matched_check = score >= confidence
            history.add((word, suggested_word, score, matched_check)) 
            # Check initials and confidence
            if (suggested_word[0] == converted_name[0]) and (score >= confidence):
                fuzzyCount += 1
                matched.add(f"{converted_name}:{suggested_word}: ratio:{score}")
                return True, suggested_word

    
            
    # if not found in history, check NER
    if ner_check_person_is_name(converted_name):
        final_words.add(converted_name)
        return True, converted_name
        
    cached_non_name_words.add(converted_name)  # Cache as non-name
    
    return False, "fail:" + word  # Ensure a tuple is always returned

def clean_ner(name):

    """Process names and avoid redundant spaCy checks."""
    name = name.strip()
    is_name_flag, returned_name = is_name(name, confidence)
    return returned_name, is_name_flag  # Return the matched name and flag


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
                #if debug_print: print(chunk)
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
    pd.DataFrame({
    "Passed Words Sorted": sorted(final_words)  # Sorted order
}).to_csv("passed_names_" + str(batch) + ".csv", index=False)
    pd.DataFrame(history, columns=["converted_name", "suggested_word", "score","matched"]).to_csv('history.csv', index=False)
    print("Passed Name Count:",len(final_words))
    print("Identified Duplicates:",len(matched))
    pd.DataFrame({"Original Names": sorted(original_names)}).to_csv("original_names.csv", index=False)

    # parse_minutes_by_id(db, 5165)
    db.close()
