import csv

# Fork from basics.py
class Item:
    # Class attributes
    payRate = 0.8
    all = []

    # A magic method to initialize class
    # With arguments specified by datatype
    # Default value for quantity = 0 (automatically specified as int)
    def __init__(self, name: str, price: float, quantity = 0):
        # Run validations to received arguments
        assert price >= 0, f'Price {price} must be at least zero!'
        assert quantity >= 0, f'Quantity {quantity} must be at least zero!'

        # Assign to self object
        self.name = name
        self.price = price
        self.quantity = quantity

        # Actions to execute
        Item.all.append(self)

    def getTotalPrice(self):
        return self.price * self.quantity
    
    def applyDiscount(self):
        # Item.payRate is class attribute, cannot be changed externally
        # self.payRate is instance attribute, can be changed externally
        self.price = self.price * self.payRate

    # Magic method to represent variable instead of
    def __repr__(self) -> str:
        return f"Item('{self.name}', {self.price}, {self.quantity})"

    # Decorator
    # cls for 'class', to distinguish from 'self
    @classmethod
    def loadFromCSV(cls):
        with open('items.csv', 'r') as f:
            reader = csv.DictReader(f)
            items = list(reader)
        
        for item in items:
            Item(
                name = item.get('name'),
                price = float(item.get('price')),
                quantity = int(item.get('quantity'))
            )

    @staticmethod
    def isInteger(number):
        if isinstance(number, float):
            return number.is_integer()
        elif isinstance(number, int):
            return True
        else:
            return False

flag_explain = True

# Basics
item1 = Item(name = "Laptop", price = 800, quantity = 2)
if flag_explain:
    print('Basics')
    print(f'Created {item1.name} with price {item1.price}, quantity {item1.quantity}.')
    print()

# Class vs. Instance attributes
# Inside class: Class.attr cannot be changed externally; instance.attr can
item2 = Item('Phone', 500, 5)
item2.payRate = 0.7
item2.applyDiscount()
if flag_explain:
    print('Class vs. Instance attributes')
    print(f'Default is Item.payRate = {Item.payRate} (Class attribute), but can be changed to {item2.payRate}.')
    print()

# Magic method __repr__ to represent variables more clearly
item3 = Item('Mouse', 20, 8)
item4 = Item('USB', 10, 5)
item5 = Item('Gamepad', 30, 2)
if flag_explain:
    print('__repr__')
    print('List of instances:')
    for instance in Item.all:
        print(f'  {instance}')
    print()

# Class method
Item.all = []
Item.loadFromCSV()    
# Static method
number = 5.0
if flag_explain:
    print('Class vs. static method')
    print(Item.all)
    print(f'Is {number} an integer: {Item.isInteger(5.0)}')