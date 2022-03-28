import requests
from bs4 import BeautifulSoup

class Scraper():
    @staticmethod
    def get_league_urls():
        url = 'https://www.skysports.com/football/tables'
        html_text = requests.get(url).text
        soup = BeautifulSoup(html_text, 'lxml')

        # Get list of leagues
        league_lis = soup.find('ul', class_ = 'page-filters__filter-body').find_all('li')

        league_urls = {}
        for li in league_lis:
            url = 'https://www.skysports.com' + li.a.attrs['href']
            league = li.a.span.span.string
            league_urls[league] = url
        return league_urls