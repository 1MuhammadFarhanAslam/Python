# OOP - Object Oriented Programming

### **1. What is OOP?**

Imagine you're building a city. In traditional, procedural programming, you'd have a massive blueprint with a long list of instructions: "Build a house, then build a road, then build a school..." If you want to add a new house, you might have to tweak many parts of that single, huge instruction list. It gets messy, hard to manage, and even harder to debug if something goes wrong. 

**Object-Oriented Programming (OOP)** is a programming **paradigm** ((a way of thinking about and structuring your code)) that centered and organizes code around **objects** rather than functions or procedures to interect with other objects when needed. Objects are entities that combine **data** (attributes) and **behavior** (methods) to model real-world concepts. These objects can contain both:

**Data**: In the form of **fields** (often known as **attributes** or **properties**).
**Code**: In the form of **procedures** (often known as **methods**).

For example:
- Imagine a **car** as an object. It has **data** (color, model, current speed, fuel level) and **behavior** (start_engine, accelerate, drive, brake, honk).
- OOP lets you create multiple car objects, each with its own data and behaviors, making your code modular and reusable.

***Think of it like building blueprints (classes) and creating actual products (objects) from them. OOP lets you model these real-world (or conceptual) things in your code.***

---

### **2. How Programming Evolved from Basic Programming to OOP?**

Programming has evolved through several stages to make code more manageable, reusable, and scalable:

1. **Machine Code (1940s-1950s)**:
   - Programs were written in binary (0s and 1s).
   - Hard to write, debug, or maintain.
   - Example: Telling a computer to add numbers required raw binary instructions.

2. **Procedural Programming (1950s-1980s)**:
   - Languages like C and Fortran introduced functions and procedures.
   - Code was organized into reusable functions, but data and functions were separate.
   - Problem: As programs grew, managing data and functions became messy. For example, in a game, you’d have separate arrays for player names, scores, and lives, and functions to manipulate them, leading to complex and error-prone code.
  
```python
# Old way: Everything mixed together like a messy kitchen
car_color = "red"
car_speed = 0
car_fuel = 100

def start_car():
    global car_speed, car_fuel
    if car_fuel > 0:
        car_speed = 10
        car_fuel -= 5

def stop_car():
    global car_speed
    car_speed = 0

# What if you want 100 cars? Copy-paste nightmare! 😱
```

3. **Object-Oriented Programming (1980s-Present)**:
   - Introduced in languages like Smalltalk, C++, and later Python.
   - OOP combines data and behavior into **objects**, making code more intuitive and modular.
   - Example: Instead of separate arrays and functions for a game, you create a `Player` object that holds its name, score, and lives, plus methods to update them.
   - Evolution driver: The need for scalable, maintainable, and reusable code for large systems (e.g., GUIs, games, enterprise software).

OOP became popular because it mirrors how we think about the real world, making it easier to model complex systems.

```python
# New way: Each car manages itself
class Car:
    def __init__(self, color):
        self.color = color
        self.speed = 0
        self.fuel = 100
    
    def start(self):
        if self.fuel > 0:
            self.speed = 10
            self.fuel -= 5
    
    def stop(self):
        self.speed = 0

# Want 100 cars? Easy! 🚗🚗🚗
my_car = Car("red")
your_car = Car("blue")
```

---

### **3. What are the Benefits of OOP?**

OOP offers several advantages that make it powerful for building robust applications:

1. **Modularity**: Objects are self-contained. The internal workings of an object can be changed without affecting other parts of the program, as long as the object's interface (how other parts interact with it) remains the same.

   - Code is organized into objects, making it easier to manage and update.
   - Example: A `Car` class can be modified without affecting other parts of the program.

2. **Reusability**: Once a class (blueprint) is created, you can create many objects (instances) from it. You can also  reuse classes in different programs (e.g., a Button class can be used in many GUI applications). This is often enhanced by Inheritance.

   - You can reuse classes in different projects or create multiple objects from the same class.
   - Example: A `Dog` class can be used to create different dogs (Labrador, Poodle) without rewriting code.
  
3. **Abstraction**: Hides complex implementation details. You interact with an object through its methods without needing to know how those methods work internally. (e.g., you press the accelerator pedal, you don't need to know the intricacies of fuel injection).

4. **Encapsulation**: Bundles data (attributes) and methods that operate on that data within a single unit (object). It can also restrict direct access to some of an object's components, preventing accidental modification (data protection).

   - Data and methods are bundled together, and access to data can be controlled (e.g., private attributes).
   - Example: A `BankAccount` object can hide its balance and only allow withdrawals through a method.

5. **Inheritance**:
   - Classes can inherit properties and methods from other classes, reducing code duplication.
   - Example: A `SportsCar` class can inherit from a `Car` class and add specialized features.

6. **Polymorphism**: (Greek for "many forms") Allows objects of different classes to respond to the same method call in their own specific way. (e.g., a Dog object and a Cat object might both have a speak() method, but one barks and the other meows).
   
   - Objects can take on multiple forms, allowing flexibility in how methods are used.
   - Example: A `Vehicle` class can have a `move` method that works differently for a `Car` or a `Boat`.

7. **Maintainability and Readability**: Code is often easier to understand, debug, and maintain because it's organized around real-world or conceptual objects. Changes to one object are less likely to break others.
   
   - OOP makes large projects easier to debug and extend because code is organized logically.
  
8. **Scalability**:
   - Want to add air conditioning to all cars? Just modify the Car class!
9. **Collaboration**: Large projects are easier to manage when different teams can work on different objects/classes independently.
---

### **4. Key Elements/Building Blocks of Python OOP**

The core building blocks of OOP in Python are:

1. **Classes**: Blueprints for creating objects. They define attributes (data) and methods (functions).
2. **Objects**: Instances of a class. Each object has its own set of attributes and can use the class’s methods.
3. **Methods**: Functions defined inside a class that describe the behaviors of an object.
4. **Attributes**: Variables that store data specific to an object or class.
5. **Abstraction**: Hides complex implementation details. You interact with an object through its methods without needing to know how those methods work internally. (e.g., you press the accelerator pedal, you don't need to know the intricacies of fuel injection).
6. **Inheritance**: A mechanism to create a new class that inherits attributes and methods from an existing class.
7. **Encapsulation**: Restricting access to certain attributes or methods to protect data.
8. **Polymorphism**: Allowing different classes to share the same method name with different implementations.

---

### **5. Classes, Objects, Methods, and Other Jargons of Python OOP**

Let’s clarify the key terms with a simple analogy:

- **Class**: A blueprint for creating objects. Think of it as a **recipe** for a cake.
- **Object**: An instance of a class. It’s the actual **cake** baked from the recipe.
- **Method**: A function defined inside a class that defines what an object can do. It’s like the **instructions** in the recipe (mix, bake).
- **Attribute**: Data stored in an object or class. It’s like the **ingredients** (flour, sugar) in the cake.
- **Instance**: Another term for an object created from a class.
- **Self**: A reference to the current object in a method. It’s like saying “this cake” when referring to the specific cake being baked.
- **Constructor**: A special method (`__init__`) that initializes a new object.
- **Inheritance**: A class (child) inheriting features from another class (parent).
- **Polymorphism**: Different classes implementing the same method differently.

**Class (The Car Blueprint):**

```python
class Car:
    # Attributes will go here
    # Methods will go here
    pass # Placeholder for now
```
**Object (A Specific Car on the Road):**

```python
my_car = Car()  # Creating an object (instance) of the Car class
your_car = Car()
```
**Attribute (Car Features/Properties):**

```python
class Car:
    def __init__(self, make, model, color):
        self.make = make
        self.model = model
        self.color = color

my_car = Car("Toyota", "Camry", "Red")
print(my_car.color) # Accessing an attribute
```
```python
class Dog:
    def __init__(self, name, breed): # __init__ is a special method
        self.name = name      # name is an instance attribute
        self.breed = breed    # breed is an instance attribute

dog1 = Dog("Buddy", "Golden Retriever")
dog2 = Dog("Lucy", "Poodle")

print(dog1.name)  # Output: Buddy
print(dog2.name)  # Output: Lucy
```


**Method (Car Actions/Behaviors):**

```python
class Car:
    def __init__(self, brand, color):  # Constructor
        self.brand = brand             # Attribute
        self.color = color

    def drive(self):                   # Method
        print(f"The {self.color} {self.brand} is driving.")

# Creating an object
my_car = Car("Toyota", "Red")
my_car.drive()  # Output: The Red Toyota is driving.
```

---

### **6. Special Methods in Classes and Why we use them?**

Special methods (also called magic methods or dunder methods) are predefined methods in Python with double underscores (__).  "Dunder" is short for "Double Under." Python calls these methods automatically in specific situations. They allow you to define how your objects behave with built-in Python operations. They give classes extra functionality.


#### **Key Special Methods**

1. **`__init__(self, ...)` The constructor method**:
   - **Purpose**: The constructor method, called when an object is created. It initializes the object’s attributes.
   - **Why use it?**: To set up initial values for an object.
   - **Example**:
     ```python
     class Car:
         def __init__(self, colour, model):
             self.colour = colour
             self.model = model
     my_car = Car("Red", "Toyota")  # __init__ sets colour and model
     ```

2. **`__str__(self)`  (String Representation for Users)**:
   - **Purpose**: Returns a human-readable string representation of the object, used by `print()` or `str()`.
   - **Why use it?**: To make objects easy to understand when printed.
   - **Example**:
     ```python
     class Car:
         def __init__(self, colour, model):
             self.colour = colour
             self.model = model
         def __str__(self):
             return f"A {self.colour} {self.model}"
     my_car = Car("Red", "Toyota")
     print(my_car)  # Output: A Red Toyota
     ```

3. **`__repr__(self)`  (Official String Representation for Developers - often seen with __str__)**:

    - **Purpose**: Called by the `repr()` built-in function. It should return a string that is an "official" or unambiguous string representation of the object. Ideally, `eval(repr(obj)) == obj`. This means the string should, if possible, be a valid Python expression that could be used to recreate the object with the same state.

    - **Analogy**: If a baker needs to write down the exact recipe and specifications to recreate this specific cookie, `__repr__` provides that detailed, unambiguous information: `Cookie(flavor='chocolate_chip', size='large', chips=25)`.

    - **Why use it?** For debugging and logging. If `__str__` is not defined, Python will fall back to using `__repr__` for print().

```python
class Car:
    def __init__(self, color, model):
        self.color = color
        self.model = model
    def __str__(self):
        return f"A beautiful {self.color} {self.model}"
    def __repr__(self):
        return f"Car(color='{self.color}', model='{self.model}')"

my_car = Car("Green", "Toyota Camry")
print(str(my_car)) # Uses __str__: A beautiful Green Toyota Camry
print(repr(my_car))# Uses __repr__: Car(color='Green', model='Toyota Camry')
print([my_car]) # Collections use __repr__ for their elements
                # Output: [Car(color='Green', model='Toyota Camry')]
```

4. **`__name__` Not a special method of a class instance in the same way, but related**:
   - **Clarification**: `__name__` is not a special method in classes but a built-in variable in Python modules. It represents the name of the module or `"__main__"` if the script is run directly.
   - **Why use it?**: To check if a Python file is being run directly or imported.
  
**For a class**: ClassName.__name__ gives you the string name of the class.

```python
class MyCoolClass:
    pass
print(MyCoolClass.__name__) # Output: MyCoolClass
```

**For a module**: When a Python file is run, `__name__` is set to `__main__` if it's the script being executed directly. If the file is imported as a module into another script, `__name__` is set to the module's filename (without .py). This is often used in the if `__name__` == `__main__`: block to write code that only runs when the file is executed directly.

```python
     if __name__ == "__main__":
         print("This script is running directly!")
```

**For functions**: function_name.`__name__` gives the string name of the function.

In OOP, you might confuse `__name__` with class-related methods, but it’s not typically used in classes.

1. **Other Common Special Methods**:
   - `__len__(self)`: Defines behavior for `len()` on an object.
   - `__eq__(self, other)`: Defines behavior for `==` comparisons.

---

### **7. Purpose of Attributes: `self.car_colour`, `self._car_colour`, `self.__car_colour`**

Attributes in Python classes store data, and their naming conventions control access and visibility. Let’s break down the differences:

1. **`self.car_colour (Public Attribute)`**:
   - **Analogy**: The car's paint job. It's visible to everyone, anyone can see it, and a body shop could (in theory) change it directly.
   - **Purpose**: This is a standard public attribute. It's meant to be accessed and modified directly from outside the class if needed.
   - **Use case**: Easily accessible No restrictions, full access from anywhere. Use for data that should be openly available
   - **Example**:

    ```python
     class Car:
        def __init__(self, colour):
        self.car_colour = colour # Public attribute

        my_car = Car("Red")
        print(my_car.car_colour)  # Accessing it directly
        my_car.car_colour = "Blue" # Modifying it directly
        print(my_car.car_colour)
    ```

2. **`self._car_colour (Protected Attribute - by convention)`**:
   - **Analogy**: The car's engine oil dipstick. While you can access it, it's generally understood that you shouldn't mess with the oil level directly unless you know what you're doing. It's more for internal checks or specialized mechanics.
   - **Purpose**: Indicates that the attribute is intended for internal use within the class or its subclasses. The single underscore is a hint to developers not to access it directly.
   - **Use case**: When you want to signal that the attribute is for internal use but still allow access if needed.
   - **Example**:
    ```python
     class Car:
        def __init__(self, colour):
            self._car_colour = colour # Protected by convention

     my_car = Car("Green")
     print(my_car._car_colour) # You can still access it
     my_car._car_colour = "Yellow" # You can still modify it (but shouldn't directly)  # Output: Red (but discouraged)
    ```

3. **`self.__car_colour (Private Attribute - by name mangling)`**:
   - **Analogy**: The car's internal engine control unit (ECU) software. You, as a driver, absolutely should not be directly fiddling with its variables. It's deeply internal, and changing it could break things badly. Only the car's own systems should interact with it directly.
   - **Purpose**: Prevents direct access from outside the class by mangling the name (e.g.`_ClassName__attribute_name`) internally. Used for strong encapsulation.
   - **Use case**: When you want to protect sensitive data and only allow access through methods.
   - **Example**:
    ```python
    class Car:
        def __init__(self, colour):
            self.__car_colour = colour # Private (name-mangled) attribute

        def get_internal_colour(self):
            return self.__car_colour

    my_car = Car("Purple")
    # print(my_car.__car_colour) # This will raise an AttributeError!
    # You would try to access it like: print(my_car._Car__car_colour)
    # But this breaks encapsulation, so you'd use a public method:
    print(my_car.get_internal_colour())  # Output: Red
     ```

**Key Difference**:
- **Public** (`car_colour`): No restrictions, anyone can access or change it.
- **Protected** (`_car_colour`): A convention to discourage direct access, but still accessible.
- **Private** (`__car_colour`): Enforces encapsulation by making the attribute harder to access directly.

---

### **8. Real-Life Simple and Easy Examples to Understand OOP**

Let’s use a **Pet Store** analogy to make OOP fun and intuitive. Imagine you’re running a pet store with dogs, cats, and birds.

#### **Example 1: Classes and Objects**
- **Class**: A blueprint for a pet, like a `Dog` class that defines what a dog is.
- **Object**: A specific dog, like a Labrador named Max.

```python
class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed
    
    def bark(self):
        return f"{self.name} says Woof!"

# Creating objects (dogs)
max_dog = Dog("Max", "Labrador")
bella_dog = Dog("Bella", "Poodle")

print(max_dog.bark())  # Output: Max says Woof!
print(bella_dog.name)  # Output: Bella
```

**Mind Map**: Think of the `Dog` class as a pet store form you fill out for each dog. Each form (object) has unique details (name, breed) but follows the same structure.

#### **Example 2: Methods and Encapsulation**
- Imagine you don’t want customers to change a dog’s name directly. You make the name **private** and provide a method to access it.

```python
class Dog:
    def __init__(self, name, breed):
        self.__name = name  # Private attribute
        self.breed = breed
    
    def get_name(self):
        return self.__name
    
    def bark(self):
        return f"{self.__name} says Woof!"

max_dog = Dog("Max", "Labrador")
print(max_dog.get_name())  # Output: Max
# print(max_dog.__name)  # Error: AttributeError
```

**Mind Map**: The private `__name` is like a dog’s ID tag locked in a safe. Only the store manager (the `get_name` method) can access it.

#### **Example 3: Inheritance**
- The pet store has a general `Pet` class, and `Dog` and `Cat` inherit from it.

```python
class Pet:
    def __init__(self, name):
        self.name = name
    
    def eat(self):
        return f"{self.name} is eating."

class Dog(Pet):  # Inherits from Pet
    def bark(self):
        return f"{self.name} says Woof!"

class Cat(Pet):  # Inherits from Pet
    def meow(self):
        return f"{self.name} says Meow!"

dog = Dog("Max")
cat = Cat("Whiskers")
print(dog.eat())  # Output: Max is eating.
print(dog.bark())  # Output: Max says Woof!
print(cat.meow())  # Output: Whiskers says Meow!
```

**Mind Map**: Think of `Pet` as a general pet care manual. `Dog` and `Cat` are specific manuals that reuse the general rules but add their own unique behaviors.

#### **Example 4: Polymorphism**
- Different pets make different sounds, but you can call a `make_sound` method on any pet.

```python
class Pet:
    def __init__(self, name):
        self.name = name
    
    def make_sound(self):
        pass  # To be overridden by subclasses

class Dog(Pet):
    def make_sound(self):
        return f"{self.name} says Woof!"

class Cat(Pet):
    def make_sound(self):
        return f"{self.name} says Meow!"

pets = [Dog("Max"), Cat("Whiskers")]
for pet in pets:
    print(pet.make_sound())
# Output:
# Max says Woof!
# Whiskers says Meow!
```

**Mind Map**: Imagine a pet show where each animal performs its unique trick (sound) when called. The `make_sound` method is like the show’s cue, but each pet responds differently.

#### **Example 5: Special Methods**
- Let’s make a `Pet` class that prints nicely with `__str__`.

```python
class Pet:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def __str__(self):
        return f"{self.name} is a {self.species}"

max_dog = Pet("Max", "Dog")
print(max_dog)  # Output: Max is a Dog
```

`__str__` is like a name tag that tells everyone what the pet is when you show it off.

---

