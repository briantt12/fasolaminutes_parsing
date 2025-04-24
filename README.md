# Sacred Harp Minutes Parser (Enhanced)

An upgraded fork of the original **Sacred Harp Minutes Parser** that leverages Large‑Language‑Models and fuzzy matching to build on original regex parser and filter out non-leader names and recognize variant spellings and aliases of singers found in Sacred Harp convention minutes from 1994 – present.

---

## ✨ What’s New

| Feature                           | Original                           | Enhanced Edition                                |
| --------------------------------- | ---------------------------------- | ----------------------------------------------- |
| **Name Extraction** | regex                        | regex + additional filtering with LLM using Ollama           |
| **Alias Matching**       | manual list of submitted names by singers | Fuzzy Matching to recognize similar names |


---

## 🚀 Quickstart
### run the scripts in this order!
`insert_minutes.py` -> (can change to current year, eg 2016)  
`create_aliases.py`  
`llm_parse_minutes.py`
`insert_locations.py`
`create_leader_stats.py`
`create_song_stats.py`
`create_song_neighbors.py`
`map_minutes_audio.py`  
`cd ./bostonsing; scrapy crawl singing`
`cd ./shapenotecds; scrapy crawl singing`
`cd ./phillysacredharp; scrapy crawl singing`
`cd ./cork; map_audio.py`
`cd ./archiveorg; map_audio.py`
`create_index.py`
