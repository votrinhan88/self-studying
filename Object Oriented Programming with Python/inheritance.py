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

# Phone (child class) inherits from Item (parent class)
class Phone(Item):
    def __init__(self, name: str, price: float, quantity = 0, broken = 0):
        # Call to parent class to have access to all attributes/methods
        super().__init__(name, price, quantity)

        # Run validations to received arguments
        assert broken >= 0, f'Number of broken phones {broken} must be at least zero!'

        # Assign to self object
        self.broken = broken
    pass

phone1 = Phone('iPhone', 700, 20, 5)
phone2 = Phone('Android', 500, 30, 10)
# Method inherited from parent class Item
print(phone1.getTotalPrice())

# 'all' is a class attribute inherited from Item to Phone
print(Item.all)
print(Phone.all)
