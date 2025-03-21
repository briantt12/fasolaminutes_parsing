# -*- coding: utf-8 -*-
import scrapy
import json

from spider_base import SpiderBase

class SingingSpider(SpiderBase):
    name = "singing"
    allowed_domains = ["shapenotecds.com"]
    # start_urls = (
    # 'http://www.shapenotecds.com/ivey-memorial-liberty-sunday-march-29-2015//',
    # )

    def parse(self, response):
        sel = scrapy.Selector(response)
        if sel.xpath('//script[@class="cue-playlist-data"]/text()'):
            self.parse_cue_playlist(response)
        elif sel.xpath('//script[@class="wp-playlist-script"]/text()'):
            self.parse_fancy(response)
        else:
            self.parse_old(response)


    def parse_old(self, response):
        sel = scrapy.Selector(response)

        sections = sel.xpath('//ol')
        for section in sections:
            song_data = []
            songs = section.xpath('./li')

            for song in songs:
                url = song.xpath('./a/@href').extract()[0]

                pagenum = None
                p = song.xpath('./a/text()').re(r'-(\d{2,3}[tb]?)')
                if p:
                    pagenum = p[0]
                else:
                    p = song.xpath('./a/text()').re(r'\s(\d{2,3}[tb]?)')
                    if p:
                        pagenum = p[0]
                    else:
                        p = song.xpath('./a/text()').re(r'(\d{2,3}[tb]?)')
                        if p:
                            pagenum = p[0]


                if not pagenum:
                    self.logger.warning("no song id: " + song.xpath('./a/text()').extract()[0])
                    continue

                song_data.append((pagenum, url))

            self.parse_section(response.url, song_data)

    def parse_fancy(self, response):
        sel = scrapy.Selector(response)

        sections = sel.xpath('//script[@class="wp-playlist-script"]/text()').extract()
        for songs in sections:
            song_data = []
            p = json.loads(songs)

            # with open(str(minutes_id) + '.json', 'w') as f:
            #     json.dump(p, f, indent=4, separators=(',', ': '))

            songs = p['tracks']
            for song in songs:
                url = song['src']
                pagenum = song['title']
                song_data.append((pagenum, url))

            self.parse_section(response.url, song_data)

    def parse_cue_playlist(self, response):
        sel = scrapy.Selector(response)

        playlist_json = sel.xpath('//script[@class="cue-playlist-data"]/text()').extract()[0]
        playlist = json.loads(playlist_json)
        songs = playlist['tracks']
        song_data = [(song['title'], song['audioUrl']) for song in songs]
        self.parse_section(response.url, song_data)

