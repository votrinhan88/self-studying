import curses
from table import Table
from typing import List

class Printer():
    def __init__(self):
        self.current_table = 0
        self.print_row = 0

    def __init_print_tables(self, tables:List[Table]):
        self.stdscr = curses.initscr()
        self.stdscr.clear()
        curses.cbreak()
        # Color
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)

        # This task could have been more efficient if executed previously in fetch_data(). But since the data is lightweight, it is kept here for readability
        len_longest_name = max([max([len(team.name)
                                        for team in table.teams])
                                            for table in tables])
                                            
        len_longest_gd  = len(str(min([min([team.gf - team.ga
                                            for team in table.teams])
                                                for table in tables])))

        self.col_2_x = {
            '#':     0,
            'Team':  4,
            'Pl':    6 + len_longest_name,
            'W':    10 + len_longest_name,
            'D':    14 + len_longest_name,
            'L':    18 + len_longest_name,
            'GD':   22 + len_longest_name,
            'GF':   24 + len_longest_name + len_longest_gd,
            'GA':   28 + len_longest_name + len_longest_gd,
            'Pt':   32 + len_longest_name + len_longest_gd,
            'Form': 36 + len_longest_name + len_longest_gd
        }
        if len(tables[0].teams[0].form) == 0:
            del(self.col_2_x['Form'])

    def __print_single_table(self, table:Table):
        name = table.name
        teams = table.teams

        # Table name & header
        self.stdscr.addstr(self.print_row, 0, name)
        for key in self.col_2_x.keys():
            self.stdscr.addstr(self.print_row + 1, self.col_2_x[key], key)
        self.print_row += 2
        # Table rows
        for row, team in enumerate(teams):
            self.stdscr.addstr(self.print_row + row, self.col_2_x['#'], str(team.pos))
            self.stdscr.addstr(self.print_row + row, self.col_2_x['Team'], team.name)
            self.stdscr.addstr(self.print_row + row, self.col_2_x['Pl'], str(team.win + team.draw + team.loss))
            self.stdscr.addstr(self.print_row + row, self.col_2_x['W'], str(team.win))
            self.stdscr.addstr(self.print_row + row, self.col_2_x['D'], str(team.draw))
            self.stdscr.addstr(self.print_row + row, self.col_2_x['L'], str(team.loss))
            self.stdscr.addstr(self.print_row + row, self.col_2_x['GD'], str(team.gf - team.ga))
            self.stdscr.addstr(self.print_row + row, self.col_2_x['GF'], str(team.gf))
            self.stdscr.addstr(self.print_row + row, self.col_2_x['GA'], str(team.ga))
            self.stdscr.addstr(self.print_row + row, self.col_2_x['Pt'], str(3*team.win + team.draw))
            # Coloring for team form
            if team.form:
                for i, result in enumerate(team.form):
                    if result == 'W':
                        color = curses.color_pair(1)
                    elif result == 'D':
                        color = curses.color_pair(2)
                    elif result == 'L':
                        color = curses.color_pair(3)
                    self.stdscr.addstr(self.print_row + row, self.col_2_x['Form'] + 2*i, result, color)
        self.print_row += len(teams)
        return

    def __print_tables(self, tables:List[Table], num_printed_tables:int):
        self.__init_print_tables(tables)
        self.print_row = 0
        for table in tables[self.current_table:self.current_table+num_printed_tables]:
            self.__print_single_table(table)
        
        # Print extra info
        self.print_row += 2
        for info in table.extra_info:
            self.stdscr.addstr(self.print_row, 0, info)
            self.print_row += 1

        # Exit
        self.stdscr.refresh()
        if len(tables) == 1:
            self.stdscr.addstr('\n\nPress ESC to exit...')
        elif len(tables) > 1:
            self.stdscr.addstr('\n\nPress A/D to scroll between tables.\nPress ESC to exit...')
    
    def print_tables(self, tables:List[Table], num_printed_tables = 2):
        self.__print_tables(tables, num_printed_tables)
        while(True):
            key = self.stdscr.getch()
            # Escape
            if key == 27: # Key Esc
                curses.endwin()
                break
            else:
                if key == 97: # Key A, move forward
                    self.current_table = (self.current_table - num_printed_tables) % len(tables)
                if key == 100: # Key D, move backward
                    self.current_table = (self.current_table + num_printed_tables) % len(tables)
                self.__print_tables(tables, num_printed_tables)
