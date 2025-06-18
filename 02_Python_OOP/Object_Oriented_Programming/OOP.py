# By default the bodies of untyped functions are not checked, consider using --check-untyped-defs  [annotation-unchecked]
# Mypy didn't check parameters of __init__ method, because it didn't have .

# class Cat:
#     def __init__(self, name, age, country):
#         self.name : str = name
#         self.age : int = age
#         self.country : str = country

# cat = Cat('Mano', 'Work', 2)
# print(cat.name)
# print(cat.age)
# print(cat.country)


##### ---------------------------------------------------------------------------------#####
# Now mypy will check type evaluation
class Cat:
    def __init__(self, name: str, age: int, country: str):
        self.name: str = name
        self.age: int = age
        self.country: str = country


cat = Cat('Mano', 2, 'Pakistan')
print(cat.name)
print(cat.age)
print(cat.country)


