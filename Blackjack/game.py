from typing import List
import numpy as np
from player import Player, Dealer

class Game():
    def __init__(self):
        # Card names
        self.card_numbers = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.card_suits = ['♠', '♣', '♦', '♥']
        self.card_names = np.array([[number + suit
                                    for suit in self.card_suits]
                                        for number in self.card_numbers], dtype = '|U3')

    # To-do: play multiple decks at one (6?) in a shoe until all cards are dealt
    #     ==> increase elements of self.deck
    #     ==> increase array dimension of self.revealed

    def add_players(self, dealer:Dealer, players:List[Player]):
        self.players = players
        self.dealer = dealer

    def __get_new_deck(self):
        # Reset state of players & dealer
        for player in self.players:
            player.reset()
        self.dealer.reset()
        # Define and shuffle deck
        self.deck = [(number, suit)
                        for suit in np.arange(len(self.card_suits))
                            for number in np.arange(len(self.card_numbers))]
        self.deck = np.random.permutation(self.deck)
        # Array of revealed cards (for easier tracking)
        self.revealed = np.zeros((len(self.card_numbers), len(self.card_suits)), dtype = bool)

    def deal_card(self, reveal_flag = True):
        # Take last card from deck
        card_deal, self.deck = self.deck[-1], self.deck[:-1]
        if reveal_flag == True:
            self.revealed[card_deal[0], card_deal[1]] = True
        return card_deal
    def put_back_card(self, card):
        # Put last card back into deck (in the last position)
        # Is the 'inverse function' of self.deal_card()
        self.deck = np.append(self.deck, [card], axis = 0)
        self.revealed[card[0], card[1]] = False

    # To-do: update rules (see note.py)
    # Dealer: + face up x 1
    # Players: + face up x 1
    # Dealer: + face down x 1
    # Players: + face up
    def play(self):
        print('New match')
        self.__get_new_deck()
        # First 2 rounds
        for round in range(2):
            for player in self.players:
                player.play()
            self.dealer.play()

        # Following rounds: each player plays until stand/bust
        # Reveal Dealer's first card to players
        print(f'Dealer: {self.card_names[self.dealer.in_hand[0][0], self.dealer.in_hand[0][1]]}')
        # Players' turn
        for player_ind, player in enumerate(self.players):  
            print(f'Player {player_ind}:')
            while player.stop_flag == False:
                player.play()
        
        # Last turn: dealer
        while self.dealer.stop_flag == False:
            self.dealer.play()

    def conclude(self):
        for player_id, player in enumerate(self.players):
            self.__evaluate(player)
            self.__transact(player)
        # Print out reports
        self.__print_report()

    def __evaluate(self, player: Player):
        # Player or Dealer has Blackjacks
        if (player.blackJack == True) & (self.dealer.blackJack == True):
            player.gain_ratio = 0
        elif (player.blackJack == True) & (self.dealer.blackJack == False):
            player.gain_ratio = 1.5
        elif (player.blackJack == False) & (self.dealer.blackJack == True):
            player.gain_ratio = -1
        # No one has Blackjacks
        elif (player.blackJack == False) & (self.dealer.blackJack == False):
            # Player busts: lost automatically
            if player.bust == True:
                player.gain_ratio = -1
            # Dealer bust, Player not: win
            elif (player.bust == False) & (self.dealer.bust == True):
                player.gain_ratio = 1
            # No one bust: compare points
            elif (player.bust == False) & (self.dealer.bust == False):
                if player.sum > self.dealer.sum:
                    player.gain_ratio = 1
                elif player.sum < self.dealer.sum:
                    player.gain_ratio = -1

    def __transact(self, player: Player):
            player.balance += player.bet * player.gain_ratio
            self.dealer.balance -= player.bet * player.gain_ratio
    
    def __print_report(self):
        print('\nGame report:')
        # Report of players
        for player_id, player in enumerate(self.players):
            player_cards = [self.card_names[card[0], card[1]]
                                for card in player.in_hand]
            
            message = f'Player {player_id}: {player_cards}'
            if player.blackJack == True:
                message += ', Blackjack'
            elif player.bust == True:
                message += ', Bust'
            else:
                message += f', Sum = {player.sum}'

            gain = player.bet * player.gain_ratio
            if gain >= 0:
                gain_str = '+' + str(gain)
            elif gain < 0:
                gain_str = str(gain)

            message += f', Balance: {player.balance} ({gain_str})'
            print(message)
        # Report of dealer
        dealer_cards = [self.card_names[card[0], card[1]]
                           for card in self.dealer.in_hand]
        message = f'Dealer: {dealer_cards}'
        if self.dealer.blackJack == True:
                message += ', Blackjack'
        elif self.dealer.bust == True:
            message += ', Bust'
        else:
            message += f', Sum = {self.dealer.sum}'
        
        gain = sum(player.bet * -player.gain_ratio for player in self.players)
        if gain >= 0:
            gain_str = '+' + str(gain)
        elif gain < 0:
            gain_str = str(gain)

        message += f', Balance: {self.dealer.balance} ({gain_str})'
        print(message)

        # print()
        # print(self.revealed.astype(int), self.revealed.sum().astype(int))
        print()