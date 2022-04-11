import numpy as np

class Player():
    def __init__(self, card_callback, balance = 300, base_bet = 10):
        self.balance = balance
        self.base_bet = base_bet
        self.card_callback = card_callback
        self.card_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    def reset(self):
        self.on_table = []              # Would need some kind of self.read()
        self.in_hand = []               # Feed into bots
        self.sum = 0                    # Feed?
        self.stop_flag = False
        self.blackJack = None
        self.bust = False
        self.can_split = None
        self.can_double = None
        self.bet = self.base_bet
        self.gain_ratio = 0

    def check_state(self):
        self.sum = self.__sum_card()
        # If black jack, return automatically (Check only once)
        if self.blackJack is None:
            if (len(self.in_hand) == 2) & (self.sum == 21):
                self.blackJack = True
                self.stop_flag = True
            else:
                self.blackJack = False

        # Can the Player double down? (Check only once)
        if self.can_double is None:
            if (len(self.in_hand) == 2) & (self.sum >= 9) & (self.sum <= 11):
                self.can_double = True
            else:
                self.can_double = False

        # Can the Player split?
        if (len(self.in_hand) == 2) & (self.in_hand[0][0] == self.in_hand[1][0]):
            self.can_split = True
        else:
            self.can_split = False

        # Else if bust, return automatically
        if len(self.in_hand) > 2:
            if self.sum > 21:
                self.bust = True
                self.stop_flag = True

    def init_brain(self):
        # self.brain
        pass

    def think(self):
        # THINK: PLUG AN BOT HERE INSTEAD OF input
        # Input: self.in_hand, self.on_table, self.sum, self.bot, self.dealer.in_hand BUT only one flag
        # Output: one action in hit() | stand() | double() | split() | ?insurance() | ?surrender()
        # Extra: Add card counting strategy
        # Extra extra: Deep learning (remove feature extraction, only input in_hand & on_table - # update memory of card on table)
        # , Double down (D) or Split (X)
        input_str = f'In hand: {self.in_hand}, Sum = {self.sum}. Hit (A), Stand (S)'
        if self.can_double == True:
            input_str += ', Double down (D)'
        if self.can_split == True:
            input_str += ', Split (X)'
        decision = input(input_str + '? ')
        decision = decision[0].lower()
        return decision

    def hit(self):          # A
        # If Player firstly hits, Doubling down and Splitting is not allowed anymore
        self.can_double = False
        self.can_split = False
        # Update memory of card in hand
        card = self.card_callback()
        self.in_hand.append(card)

    def stand(self):        # S
        self.stop_flag = True
    
    def double(self):       # D
        # Doubling their bet when the original two cards dealt total 9, 10, or 11. When the player's turn comes, they place a bet equal to the original bet, and the dealer gives the player just one card, which is placed face down and is not turned up until the bets are settled at the end of the hand. Note that the dealer does not have the option of splitting or doubling down.
        self.can_double = False
        self.bet *= 2
        card = self.card_callback()
        self.in_hand.append(card)
        # After picking one more card, Player's turn is over
        self.stop_flag = True

    def split(self):        # X
    # If a player's first two cards are of the same denomination, such as two jacks or two sixes, they may choose to treat them as two separate hands when their turn comes around. The amount of the original bet then goes on one of the cards, and an equal amount must be placed as a bet on the other card. The player first plays the hand to their left by standing or hitting one or more times; only then is the hand to the right played. The two hands are thus treated separately, and the dealer settles with each on its own merits. With a pair of aces, the player is given one card for each ace and may not draw again. Also, if a ten-card is dealt to one of these aces, the payoff is equal to the bet (not one and one-half to one, as with a blackjack at any other time).
        # self.can_split = False
        print('Splitted')
        self.reserved_hands = []
        self.reserved_hands.append([self.in_hand.pop()])

    def play(self):
        # Request until player has 2 cards (Note: != hit())
        if len(self.in_hand) < 2:
            card = self.card_callback()
            self.in_hand.append(card)
            return

        elif len(self.in_hand) >= 2:
            # Check for Blackjack or Bust (stop automaticallly) 
            self.check_state()
            if self.stop_flag == True:
                return

            # Will plug an AI bot into here (right now through console input)
            decision = self.think()
            # Making a move: hit, stand, double down, or split
            if decision == 'a':
                self.hit()
            elif decision == 's':
                self.stand()
            elif decision == 'd':
                self.double()
            elif decision == 'x':
                self.split()
            
    def __sum_card(self):
        # Check for Aces in hand
        aces = []
        non_aces = []
        # No aces in hand
        for index, card in enumerate(self.in_hand):
            if card[0] == 0:
                aces.append(card)
            else:
                non_aces.append(card)

        num_ace = len(aces)
        if num_ace == 0:
            sum = np.sum(self.card_value[card[0]] for card in self.in_hand)
        elif num_ace >= 1:
            sum_non_aces = np.sum(self.card_value[card[0]] for card in non_aces)

            sum_aces = np.zeros(shape = ((2,)*3), dtype = np.int8)
            ace_values = np.array([1, 11], dtype = np.int8)
            for dim in range(num_ace):
                sum_aces[:, :, :] += ace_values
                ace_values = np.expand_dims(ace_values, axis = 1)
            sum = sum_aces + sum_non_aces
        
            # Check if the play is bust (<= 21) or not
            # Note: in case the player does not have aces, this step is unnecessary
            # because sum is fixed
            min_sum = np.min(sum)
            # Bust: sum is min value
            if min_sum > 21:
                sum = np.min(sum)
            # Not-bust: sum is max value which yields <= 21
            elif min_sum <= 21:
                sum = np.max(sum[sum <= 21])
        if sum > 21:
            self.bust = True
        return sum


class Dealer():
    def __init__(self, card_callback, balance = 1000):
        self.card_callback = card_callback
        self.balance = balance

        self.card_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    
    def reset(self):
        self.on_table = []
        self.in_hand = []
        self.sum = 0
        self.stop_flag = False
        self.blackJack = None
        self.bust = False
    
    def check_state(self):
        self.sum = self.__sum_card()
        # If black jack, return automatically (Check only once)
        if self.blackJack == None:
            if (len(self.in_hand) == 2) & (self.sum == 21):
                self.blackJack = True
                self.stop_flag = True
            else:
                self.blackJack = False

        # If bust, return automatically
        elif len(self.in_hand) > 2:
            if self.sum > 21:
                self.bust = True
                self.stop_flag = True

    def hit(self):
        card = self.card_callback()
        # Update memory of card on table and card in hand
        self.in_hand.append(card)
        self.sum = self.__sum_card()

    def play(self):
        self.check_state()
        if self.stop_flag == True:
            return
        
        # Dealer cannot think, but play automatically.
        # If the total is 17 or more, it must stand.
        if self.sum >= 17:
            self.stop_flag = True
        # If the total is 16 or under, they must take a card.
        if self.stop_flag == False:
            self.hit()
    
    def __sum_card(self):
        # Check for Aces in hand
        aces = []
        non_aces = []
        # No Ace (A) in hand
        for card in self.in_hand:
            if card[0] == 0:
                aces.append(card)
            else:
                non_aces.append(card)

        num_ace = len(aces)
        if num_ace == 0:
            sum = np.sum(self.card_value[card[0]] for card in self.in_hand)
        elif num_ace >= 1:
            sum_non_aces = np.sum(self.card_value[card[0]] for card in non_aces)

            sum_aces = np.zeros(shape = ((2,)*3), dtype = np.int8)
            ace_values = np.array([1, 11], dtype = np.int8)
            for dim in range(num_ace):
                sum_aces[:, :, :] += ace_values
                ace_values = np.expand_dims(ace_values, axis = 1)
            sum = sum_aces + sum_non_aces
        
            # Check if the play is bust (<= 21) or not
            # Note: in case the player does not have aces, this step is unnecessary
            # because sum is fixed
            min_sum = np.min(sum)
            # Bust: sum is min value
            if min_sum > 21:
                sum = np.min(sum)
            # Not-bust: sum is max value which yields <= 21
            elif min_sum <= 21:
                sum = np.max(sum[sum <= 21])
        # if sum > 21:
        #     self.bust = True
        return sum