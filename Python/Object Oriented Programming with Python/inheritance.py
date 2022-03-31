# Inheritance: allow reuse of code across classes (downstream to child classes)

from classes.item import Item
from classes.phone import Phone

phone1 = Phone('iPhone', 700, 20, 5)
phone2 = Phone('Android', 500, 30, 10)

# Method inherited from parent class Item
print(phone1.getTotalPrice())

# 'all' is a class attribute inherited from Item to Phone
print(Item.all)
print(Phone.all)
