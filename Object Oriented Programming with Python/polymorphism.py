# Polymorphism (Many-forms): the use of a single type entity to represent different types in different scenarios

# len() knows how to handle different kinds of objects
name = "Nathan" 
list = [1, 2, 3, 4]
print(len(name))
print(len(list))

# applyDiscount() handling can be adjusted with a payRate different from default
from classes.laptop import Laptop
item1 = Laptop('Gaming Laptop', 1500, 2)
item1.applyDiscount()
print(item1.price)