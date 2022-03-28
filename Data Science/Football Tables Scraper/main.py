from bs4 import BeautifulSoup
import requests
import re
import curses
from team import Team

class Table():
    teams = []
    
    @classmethod
    def fetch_data(cls, table_url = 'https://www.skysports.com/premier-league-table'):
        html_text = requests.get(table_url).text
        soup = BeautifulSoup(html_text, 'lxml')
        main_table = soup.find(name = 'table', class_ = 'standing-table__table')
        
        # Each row = team
        for table_row in main_table.tbody.find_all(name = 'tr'):
            table_data = table_row.find_all(name = 'td')

            team = Team(
                pos  = int(table_data[0].string),
                name = table_data[1].a.string,
                win  = int(table_data[3].string),
                draw = int(table_data[4].string),
                loss = int(table_data[5].string),
                gf   = int(table_data[6].string),
                ga   = int(table_data[7].string),
                # span.attrs['class'][1] = 'standing-table__form-cell--[win|draw|loss]'
                # ==> Fetch list of 'W', 'D' or 'L'
                form = [span.attrs['class'][1][27].upper() for span in table_data[10].div.find_all('span')]
            )
            cls.teams.append(team)
        return cls.teams

    @classmethod
    def print_table(cls):
        stdscr = curses.initscr()
        # Color
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)

        # This task could have been more efficient if executed previously in fetch_data(). But since the data is lightweight, it is kept here for readability
        len_longest_name = max([len(team.name) for team in cls.teams])

        col_2_x = {
            '#':    0,
            'Team': 4,
            'Pl':    6  + len_longest_name,
            'W':    10 + len_longest_name,
            'D':    14 + len_longest_name,
            'L':    18 + len_longest_name,
            'GD':   22 + len_longest_name,
            'GF':   26 + len_longest_name,
            'GA':   30 + len_longest_name,
            'Form': 34 + len_longest_name
        }

        # Table header
        for key in col_2_x.keys():
            stdscr.addstr(0, col_2_x[key], key)

        while(True):
            for row, team in enumerate(cls.teams):
                stdscr.addstr(row + 1, col_2_x['#'], str(team.pos))
                stdscr.addstr(row + 1, col_2_x['Team'], team.name)
                stdscr.addstr(row + 1, col_2_x['Pl'], str(team.win + team.draw + team.loss))
                stdscr.addstr(row + 1, col_2_x['W'], str(team.win))
                stdscr.addstr(row + 1, col_2_x['D'], str(team.draw))
                stdscr.addstr(row + 1, col_2_x['L'], str(team.loss))
                stdscr.addstr(row + 1, col_2_x['GD'], str(team.gf - team.ga))
                stdscr.addstr(row + 1, col_2_x['GF'], str(team.gf))
                stdscr.addstr(row + 1, col_2_x['GA'], str(team.ga))
                # Coloring for team form
                for i, result in enumerate(team.form):
                    if result == 'W':
                        color = curses.color_pair(1)
                    elif result == 'D':
                        color = curses.color_pair(2)
                    elif result == 'L':
                        color = curses.color_pair(3)
                    stdscr.addstr(row + 1, col_2_x['Form'] + 2*i, result, color)
            stdscr.addstr('\n\nPress ESC to exit...')
            stdscr.refresh()
            # Escape
            if stdscr.getch() == 27:
                curses.endwin()
                break

Table.fetch_data()
Table.print_table()