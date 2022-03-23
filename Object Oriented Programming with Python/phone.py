import csv
from item import Item

# Fork from inheritance.py
# Phone (child class) inherits from Item (parent class)
class Phone(Item):
    def __init__(self, name: str, price: float, quantity = 0, broken = 0):
        # Call to parent class to have access to all attributes/methods
        super().__init__(name, price, quantity)

        # Run validations to received arguments
        assert broken >= 0, f'Number of broken phones {broken} must be at least zero!'

        # Assign to self object
        self.broken = broken