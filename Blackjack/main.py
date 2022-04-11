from player import Player, Dealer
from game import Game



game = Game()

players = [Player(card_callback = game.deal_card),
           Player(card_callback = game.deal_card),
           Player(card_callback = game.deal_card)]
dealer = Dealer(card_callback = game.deal_card)
game.add_players(dealer, players)

try:
    while True:
        game.play()
        game.conclude()
except KeyboardInterrupt:
    pass
