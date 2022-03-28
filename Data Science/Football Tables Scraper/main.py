from bs4 import BeautifulSoup
import requests
import re
import curses

class Team():
    def __init__(self, name:str, pos:int, prev:int, win:int, draw:int, loss:int, gf:int, ga:int, form:list):
        self.name = name
        self.pos  = pos
        self.prev = prev
        self.win  = win 
        self.draw = draw
        self.loss = loss
        self.gf   = gf  
        self.ga   = ga  
        self.form = form
    
class Table():
    teams = []

    @classmethod
    def fetch_data(cls):
        html_text = requests.get('https://www.premierleague.com/tables').text
        soup = BeautifulSoup(html_text, 'lxml')
        main_content = soup.find(name = 'tbody', class_ = 'tableBodyContainer isPL')
        rows = main_content.find_all(name = 'tr')

        for row in rows:
            if 'expandable' not in row.attrs['class']:
                team_data = row.find_all(name = 'td')
                cls.teams.append(Team(
                    name = team_data[2].find('span', class_ = 'long').string,
                    pos  = int(team_data[1].find('span', class_ = 'value').string),
                    prev = int(re.sub('[^0-9]', '', team_data[1].find('span', class_ = 'resultHighlight').string)),
                    win  = int(team_data[4].string),
                    draw = int(team_data[5].string),
                    loss = int(team_data[6].string),
                    gf   = int(team_data[7].string),
                    ga   = int(team_data[8].string),
                    form = [result.abbr.string for result in team_data[11].find_all('li')]))
        return cls.teams

    @classmethod
    def print_table(cls):
        stdscr = curses.initscr()
        # Color
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)

        stdscr.addstr(0, 0, 'Δ')
        stdscr.addstr(0, 4, '#')
        stdscr.addstr(0, 8, 'Team')
        stdscr.addstr(0, 34, 'P')
        stdscr.addstr(0, 38, 'W')
        stdscr.addstr(0, 42, 'D')
        stdscr.addstr(0, 46, 'L')
        stdscr.addstr(0, 50, 'GD')
        stdscr.addstr(0, 54, 'GF')
        stdscr.addstr(0, 58, 'GA')
        stdscr.addstr(0, 62, 'Form')

        while(True):
            for row, team in enumerate(cls.teams):
                if team.pos > team.prev:
                    delta = '↑'
                    color = curses.color_pair(1)
                elif team.pos == team.prev:
                    delta = '-'
                    color = curses.color_pair(2)
                elif team.pos < team.prev:
                    delta = '↓'
                    color = curses.color_pair(3)
                stdscr.addstr(row + 1, 0, delta, color)

                stdscr.addstr(row + 1, 4, str(team.pos))
                stdscr.addstr(row + 1, 8, team.name)
                stdscr.addstr(row + 1, 34, str(team.win + team.draw + team.loss))
                stdscr.addstr(row + 1, 38, str(team.win))
                stdscr.addstr(row + 1, 42, str(team.draw))
                stdscr.addstr(row + 1, 46, str(team.loss))
                stdscr.addstr(row + 1, 50, str(team.gf - team.ga))
                stdscr.addstr(row + 1, 54, str(team.gf))
                stdscr.addstr(row + 1, 58, str(team.ga))

                for i, result in enumerate(team.form):
                    if result == 'W':
                        color = curses.color_pair(1)
                    elif result == 'D':
                        color = curses.color_pair(2)
                    elif result == 'L':
                        color = curses.color_pair(3)
                    stdscr.addstr(row + 1, 62 + 2*i, result, color)

            stdscr.refresh()
            # Escape
            if stdscr.getch() == 27:
                curses.endwin()
                break

Table.fetch_data()
Table.print_table()