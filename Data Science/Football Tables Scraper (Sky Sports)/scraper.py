import requests
from bs4 import BeautifulSoup
import re
from table import Table, Team

class Scraper():
    def get_league_urls(self):
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

    def fetch_data(self, league_url:str):
        # Scrape table
        full_page = requests.get(league_url).text
        soup = BeautifulSoup(full_page, 'lxml')
        full_tables = soup.find_all('div', class_ = 'standing-table standing-table--full block')
        tables = []
        for full_table in full_tables:        
            # Table name (and trim whitespace at beginning and end)
            table_name = re.sub('^[ \t\r\n]+|[ \t\r\n]+$', '', full_table.table.caption.string)
            # Extra info (and regex)
            table_suppl = full_table.find('div', class_ = 'standing-table__supplementary').children
            extra_info = []
            for child in table_suppl:
                text_regex = re.sub('<[^>]*>|\n', '', str(child))
                # ['\n', 'Last updated: March 21, 2022 2:53am', 'Key', 'Uefa Champions League: 1st, 2nd, 3rd, 4th', '', 'Europa League: 5th', '', 'Relegation: 18th, 19th, 20th']
                if text_regex != '':
                    if text_regex == 'Key':
                        extra_info.append('')
                    else:
                        extra_info.append(text_regex)
            
            # Table rows (each row is one team)
            teams = []
            for table_row in full_table.table.tbody.find_all(name = 'tr'):
                table_data = table_row.find_all(name = 'td')
                # Fetch form (is structured differently)
                form = []
                if table_data[10].div != None:
                    for span in table_data[10].div.find_all('span'):
                        form.append(span.attrs['class'][1][27].upper())
                # Fetch team data from table cells
                team = Team(
                    pos  = int(table_data[0].string),
                    name = table_data[1].find(re.compile('(a|span)')).string,
                    win  = int(table_data[3].string),
                    draw = int(table_data[4].string),
                    loss = int(table_data[5].string),
                    gf   = int(table_data[6].string),
                    ga   = int(table_data[7].string),
                    # span.attrs['class'][1] = 'standing-table__form-cell--[win|draw|loss]'
                    # ==> Fetch list of 'W', 'D' or 'L'
                    form = form
                )
                teams.append(team)
            
            table = Table(
                name = table_name,
                teams = teams,
                extra_info = extra_info)
            tables.append(table)
        return tables