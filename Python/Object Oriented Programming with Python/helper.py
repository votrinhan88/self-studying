# When to use class methods OR static methods?

class Item:
    @staticmethod
    def someStaticMethod():
        '''
        This should do something that has a relationship with the class, but not something that must be unique per instance.
        It does not require a variable.
        Can stand alone as an individual function --> put on top (close to outside of class).
        Example: isInteger()
        '''
        pass
    
    @classmethod
    def someClassMethod(cls):
        '''
        This should do something that has a relationship with the class, but usually to manipulate different structures of data to instantitate objects.
        It requires a mandatory variable (cls of class, different from self of internal methods).
        Example: loadFromCSV()
        '''
        pass

'''Class methods and static methods can also be called from instances; but they never should be to avoid confusion'''
item = Item()
# DON'T: Valid but confusing
item.someClassMethod() 
item.someStaticMethod()
# DO:
Item.someClassMethod()
Item.someStaticMethod()