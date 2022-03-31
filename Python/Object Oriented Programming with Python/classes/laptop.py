from .item import Item

# Laptop (child class) inherits from Item (parent class)
class Laptop(Item):
    payRate = 0.7

    def __init__(self, name: str, price: float, quantity = 0, broken = 0):
        # Call to parent class to have access to all attributes/methods
        super().__init__(name, price, quantity)