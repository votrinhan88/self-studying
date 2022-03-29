# Snippet: Change of position (colored)
# Depreciated because tables from new datasource does not have previous position (team.prev)
'''
class Table():
    ...
    def print_table(cls):
        ...
        while(True):
            ...
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
            ...
'''