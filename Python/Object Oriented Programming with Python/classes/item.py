import csv

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
        self.__name = name
        self.__price = price
        self.quantity = quantity

        # Actions to execute
        Item.all.append(self)

    # @property decorator = read-only attribute
    @property
    def name(self):
        return self.__name
    @property
    def price(self):
        return self.__price
    def applyDiscount(self):
        # Item.payRate is class attribute, cannot be changed externally
        # self.payRate is instance attribute, can be changed externally
        self.__price = self.__price * self.payRate
    def applyIncrement(self, increment = 0.1):
        self.__price = self.__price * (1 + increment)
        
    # @setter decorator = reverse read-only attribute
    @name.setter
    def name(self, value):
        if len(value) > 10:
            raise Exception('The name cannot have more than 10 characters.')
        else:
            self.__name = value

    def getTotalPrice(self):
        return self.__price * self.quantity

    # Magic method to represent variable instead of
    def __repr__(self) -> str:
        # Magic attribute to get class name
        return f"{self.__class__.__name__}('{self.name}', {self.price}, {self.quantity})"

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