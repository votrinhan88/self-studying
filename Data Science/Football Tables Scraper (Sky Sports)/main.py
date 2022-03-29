from printer import Printer
from scraper import Scraper

# Get available leagues
league_urls = Scraper().get_league_urls()
list_leagues = [*league_urls]

# List the leagues to choose from
print('List of available leagues')
for index, league in enumerate(list_leagues):
    print(f'[{index}]\t{league}')
selected = input('Choose league by number, or type any letter to exit: ')
if selected.isnumeric():
    # Fetch league data and print table(s)
    tables = Scraper().fetch_data(league_urls[list_leagues[int(selected)]])
    Printer().print_tables(tables = tables, num_printed_tables = 2)
else:
    print('Thank you')