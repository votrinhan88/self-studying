from pick import pick
from league import League
from scraper import Scraper
import curses

league_urls = Scraper.get_league_urls()
title = 'Please choose a league ...\nChange option with J/K and select with Enter\nPress ESC to exit...'
options = list(league_urls.keys())
selected, _ = pick(options, title, indicator = '=>')
League.fetch_data(league_urls[selected])
League.print_table()