# Encapsulation: restrict the ability to override instance values

from classes.item import Item

item1 = Item('Bicycle', 1000)

# Will raise error
if False:
    # Cannot access attribute
    print(item1.__name)
    print(item1.__price)
    # Cannot set attribute
    item1.price = 200

    # With @setter, can set attribute; but with restrictions
    # Exception: Name > 10 characters
    item1.name = 'Motorbike NEW'

item1.name = 'Motorbike'
item1.applyIncrement(0.2)
item1.applyDiscount()

print(item1.name)
print(item1.price)