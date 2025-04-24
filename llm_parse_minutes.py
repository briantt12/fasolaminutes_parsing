#!/usr/bin/env python
# encoding: utf-8

import re
import util
import openai
import os
import ollama
import pandas as pd
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt
import time

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

cached_non_name_words = set()  # Cache of confirmed non-names
cached_name_words = set()  # Cache of confirmed names
batch = 1000
confidence = 87

times=[]
names_count = []
passed_names = set()  # Store full names for fuzzy matching
matched = set()
final_words = set()
failed = set()
fuzzyCount = 0
start_time = time.time()
history = set()
original_names = set()
matched_alias = {}
time_counts = [(0,0)]

def is_valid_name(candidate, context):
    """
    Uses an LLM (via Ollama) to determine whether a candidate is a valid leader name.
    
    Parameters:
    - candidate (str): The extracted name.
    - context (str): The surrounding text in the minutes for better accuracy.

    Returns:
    - bool: True if the candidate is a valid leader's name, False otherwise.
    """
    prompt = f"""
    You are analyzing meeting minutes to extract leader names.
    Given the following context:
    "{context}"

    Does "{candidate}" appear to be a personal name (not a role or organization)? 
    You should only respond "Yes" for names of people who lead songs at Sacred Harp singings.
    If they are song writers or included in names of places for instance, you should respond "No".
    Respond with "Yes" or "No" only.
    """
    if candidate in cached_name_words:
        return True
    if candidate in cached_non_name_words:
        return False

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}], options={"num_ctx": 32768})
    
    is_valid = response["message"]["content"].strip().lower() == "yes"
    
    if is_valid:
        cached_name_words.add(candidate)
    else:
        cached_non_name_words.add(candidate)
    

    return is_valid

def clean_ner(name, context):
    """Process names and avoid redundant checks using is_valid_name."""
    name = name.strip()
    is_name_flag = is_name(name, confidence, context)  # Use the LLM-based name validation
    return name, is_name_flag  # Return the processed name and flag

excluded_abbreviated = {'USA.'}

def standardize_name(name,case):
    """Convert the second name to an initial if there are more than two words."""
    original_names.add(name)
    words = name.split()
    if words[0].upper() in excluded_abbreviated:
        words = words[1:]  # remove it
    if case ==1:
    # if '-' in name:
        firstname = words[0]
        lastname = name.split('-')[-1]
        return firstname+' '+lastname
    elif case == 2:
    # if len(words) >= 3:  # If there's a middle name or extra names
        return f"{words[0]} {words[-1]}"  # Convert only the second word to an initial

    
    elif case == 3 and len(words) > 1:
        # Extract first name and first part of hyphenated last name
        firstname = words[0]
        lastname_first_half = words[1].split('-')[0]
        return firstname + ' ' + lastname_first_half
    elif case == 4 and len(words)==3:
        return f"{words[0]} {words[1]}-{words[2]}" #try hyphenated name
    else:
        return name  # Return as is if single word

def plot_pts():
    global start_time, time_counts
   
    if(len(final_words)% batch==0):
        elapsed_time = time.time() - start_time
        time_counts.append((elapsed_time,len(final_words)))
        df = pd.DataFrame(time_counts, columns=['time', 'count'])
        df.to_csv('timeDataSpacyEnhanced.csv', index=False)
        
def convert_to_initials(name):
    """Convert the second name to an initial if there are more than two words."""
    words = name.split()
    if len(words) >= 3:  # If there's a middle name or extra names
        return f"{words[0]} {words[1][0]}. {words[-1]}"  # Convert only the second word to an initial
    if len(words) == 2:
        return name  # Keep first and last name as is
    return name  # Return as is if single word

def is_name(word,confidence,context):
    global fuzzyCount
    plot_pts()

    
        
    
    
    
    if word in final_words:
        return True, word

    elif word in cached_non_name_words:
        return False, "fail:" + word  # Cached as non-name

    if '-' in word:
        converted_name = standardize_name(word,1) # was 4
        if converted_name in final_words:
            return True, converted_name
        ratio_match = process.extractOne(converted_name, final_words, scorer=fuzz.ratio)
        
        if ratio_match:
            suggested_word, score, _ = ratio_match
            matched_check = score >= confidence
            history.add((converted_name, suggested_word, score, matched_check)) 
            # Check initials and confidence
            if (suggested_word[0] == converted_name[0]) and (score >= confidence):
                fuzzyCount += 1
                matched.add(f"{converted_name}:{suggested_word}: ratio:{score}")
                if suggested_word not in matched_alias:
                    matched_alias[suggested_word] = []
                if word not in matched_alias[suggested_word]:
                    matched_alias[suggested_word].append(word)
                return True, suggested_word
            
        converted_name = standardize_name(word,3) #was 1
        if converted_name in final_words:
            return True, converted_name
        ratio_match = process.extractOne(converted_name, final_words, scorer=fuzz.ratio)
        
        if ratio_match:
            suggested_word, score, _ = ratio_match
            matched_check = score >= confidence
            history.add((converted_name, suggested_word, score, matched_check)) 
            # Check initials and confidence
            if (suggested_word[0] == converted_name[0]) and (score >= confidence):
                fuzzyCount += 1
                matched.add(f"{converted_name}:{suggested_word}: ratio:{score}")
                if suggested_word not in matched_alias:
                    matched_alias[suggested_word] = []
                if word not in matched_alias[suggested_word]:
                    matched_alias[suggested_word].append(word)
                
                return True, suggested_word
        
    
    if len(word.split()) >= 3:
        converted_name = standardize_name(word,4)
        # print('occurred:', converted_name)
        if converted_name in final_words:
            return True, converted_name
        
        # Fuzzy match on shortened name
        ratio_match = process.extractOne(converted_name, final_words, scorer=fuzz.ratio)
        if ratio_match:
            suggested_word, score, _ = ratio_match
            matched_check = score >= confidence
            history.add((converted_name, suggested_word, score, matched_check)) 
            
            if (suggested_word[0] == converted_name[0]) and (score >= confidence):
                fuzzyCount += 1
                matched.add(f"{converted_name}:{suggested_word}: ratio:{score}")
                if suggested_word not in matched_alias:
                    matched_alias[suggested_word] = []
                if word not in matched_alias[suggested_word]:
                    matched_alias[suggested_word].append(word)
                return True, suggested_word
            
        
        converted_name = standardize_name(word,2)
        if converted_name in final_words:
            return True, converted_name
        
        # Fuzzy match on shortened name
        ratio_match = process.extractOne(converted_name, final_words, scorer=fuzz.ratio)
        if ratio_match:
            suggested_word, score, _ = ratio_match
            matched_check = score >= confidence
            history.add((converted_name, suggested_word, score, matched_check)) 
            
            if (suggested_word[0] == converted_name[0]) and (score >= confidence):
                fuzzyCount += 1
                matched.add(f"{converted_name}:{suggested_word}: ratio:{score}")
                if suggested_word not in matched_alias:
                    matched_alias[suggested_word] = []
                if word not in matched_alias[suggested_word]:
                    matched_alias[suggested_word].append(word)
                return True, suggested_word


    if len(word.split()) < 3:
        converted_name = standardize_name(word,6)
        if converted_name in final_words:
            return True, converted_name
        ratio_match = process.extractOne(converted_name, final_words, scorer=fuzz.ratio)
        
        if ratio_match:
            suggested_word, score, _ = ratio_match
            matched_check = score >= confidence
            history.add((converted_name, suggested_word, score, matched_check)) 
            # Check initials and confidence
            if (suggested_word[0] == converted_name[0]) and (score >= confidence):
                fuzzyCount += 1
                matched.add(f"{converted_name}:{suggested_word}: ratio:{score}")
                if suggested_word not in matched_alias:
                    matched_alias[suggested_word] = []
                if word not in matched_alias[suggested_word]:
                    matched_alias[suggested_word].append(word)
                return True, suggested_word
    

    
            
    # if not found in history
    if is_valid_name(word, context):
        final_words.add(word)
        if word not in matched_alias:
            matched_alias[word] = []
        if word not in matched_alias[word]:
            matched_alias[word].append(word)
        return True, word
        
    cached_non_name_words.add(word)  # Cache as non-name
    
    return False, "fail:" + word  # Ensure a tuple is always returned


def expand_family_names(candidate):
    """
    Checks if the candidate matches a family/group entry of the form:
      e.g. "Brock Family (Christy, Bradley, Jonah, Maribelle)",
           "Smith Sisters (Sarah, Sara)", or 
           "Tekadtuera Siblings (Brian, Aman)"
    
    If a match is found, it extracts the root name (e.g. "Brock", "Smith", "Tekadtuera")
    and returns a list of individual names with each first name appended to that root.
    Otherwise, it returns None.
    """
    # This regex matches a word, followed by one of the keywords, then a parenthesized list.
    match = re.search(r'(\w+)\s+(Family|Sisters|Brothers|Siblings)\s*\(([^)]+)\)', candidate)
    if match:
        root_name = match.group(1)
        # group(3) contains the comma-separated first names inside the parentheses
        first_names = [n.strip() for n in match.group(3).split(',')]
        expanded = [f"{first} {root_name}" for first in first_names]
        print(expanded)
        return expanded
    return None

# Updated parse_minutes fuction for when expand_family_names is able to be fully implemented
"""
def parse_minutes(s, debug_print=False):
    session_count = 0
    sessions = re.split('RECESS|LUNCH', s)
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
        leaders = re.split(r'\v|called to order|\:\s|(?<=[^\.][^A-Z\]\}])\.(\s|\Z)|(?<=[\]\}”\)])[;\.\:]|;', session)  # double quotes!
        for chunk in leaders:
            if chunk and (len(chunk) > 2):
                if debug_print:
                    print(chunk)
                songs = re.finditer(pagenum_pattern, chunk)
                first_song = None
                for song in songs:
                    if not first_song:
                        first_song = song
                    pagenum = song.group(1)
                    # print pagenum
                    leaders = re.finditer(name_pattern, chunk)
                    for leader in leaders:
                        if leader.end() <= first_song.start() + 1:
                            raw_name = leader.group(0).strip()  # TODO: should be able to incorporate this into regex......
                            # Attempt to expand the name if it's a family/group entry (Family, Sisters, Brothers, Siblings)
                            expanded = expand_family_names(raw_name)
                            if expanded:
                                for ename in expanded:
                                    print(ename)
                                    cached_name_words.add(ename)
                                    final_words.add(ename)
                                    dd.append({'name': ename, 'song': pagenum})
                                    if debug_print:
                                        print('***name: ' + ename + '\tsong: ' + pagenum)
                            else:
                                name, flag = clean_ner(raw_name, session)
                                if flag == True:                                
                                    dd.append({'name': name, 'song': pagenum})
                                if debug_print:
                                    print('***name: ' + name + '\tsong: ' + pagenum)
                        # else:
                            # print "%d %d"%(leader.end(), first_song.start())
                if debug_print:
                    print("---chunk----------")

        d.append({'session': session_count, 'leaders': dd})
        # print "---session----------"
    # print d
    return d
"""

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
                            name, flag = clean_ner(name, session)
                            if flag[0] == True:                                
                                dd.append({'name': flag[1], 'song': pagenum})
                            if debug_print: print('***name: ' + name + '\tsong: ' + pagenum)
                        # else:
                            # print "%d %d"%(leader.end(), first_song.start())
                if debug_print: print("---chunk----------")
        d.append({'session': session_count, 'leaders': dd})
        # print "---session----------"
    # print d
    return d

def parse_all_minutes_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Split line into fields (assuming '|' as delimiter)
            row = line.strip().split("|")  

            minutes_text = row[0]
            
            parse_minutes(minutes_text)


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
    pd.DataFrame({"Name Words": list(cached_name_words)}).to_csv("llm_modified_names.csv", index=False)
    pd.DataFrame({"Non-Name Words": list(cached_non_name_words)}).to_csv("llm_modified_non_names.csv", index=False)
    print("name count:",len(cached_name_words))
    print("non name count:",len(cached_non_name_words))
    # parse_minutes_by_id(db, 5165)
    db.close()
    '''
    parse_all_minutes_from_file("test_minutes.txt")
    pd.DataFrame({"Name Words": list(cached_name_words)}).to_csv("llm_modified_names.csv", index=False)
    pd.DataFrame({"Non-Name Words": list(cached_non_name_words)}).to_csv("llm_modified_non_names1.csv", index=False)
    pd.DataFrame({"Matched Words": sorted(list(matched))}).to_csv("matched_names.csv", index=False)
    final_words.update(passed_names)
    pd.DataFrame({
    "Passed Words Ordered": list(final_words),  # Original order
    "Passed Words Sorted": sorted(final_words)  # Sorted order
}).to_csv("passed_names_" + str(batch) + ".csv", index=False)
    print("Passed Name Count:",len(final_words))
    print("Identified Duplicates:",len(matched))
    '''
