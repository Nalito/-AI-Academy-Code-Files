# Data Structures in python

## Lists
ages = [12, 34, 45, 66, 32, 25, 45, 76]
print(ages)

# Access the first element
print(ages[0])

# Access the last element
print(ages[-1])

# Subset a list
print(ages[2:4])

# Pop from a list
print(ages.pop())
print(ages)

# Delete from a list using a specific index
del ages[0]

# Append to a list
ages.append(100)
print(ages)

# Extend a list
ages.extend([65, 88])
print(ages)

# Insert into a list
ages.insert(3, 55)
print(ages)

## Dictionaries
my_dict = {'name': 'James Bond', 'age': 55, 'contact': ['jb@hotmail.com', 55545898]}
print(my_dict)

# Access the first element
print(my_dict['name'])

# Access the last element
print(my_dict['contact'])

# Accessing elements using .get()
print(my_dict.get('age'))

# Fetch the keys from a dictionary
print(my_dict.keys())

# Fetch the values from a dictionary
print(my_dict.values())

# Fetch the key-value pairs from a dictionary
print(my_dict.items())

# Add to a dictionary
my_dict['is_student'] = False
print(my_dict)

# Delete from a dictionary


## Tuples
tup = ('introduction', 'to', 'artificial', 'intelligence')
print(tup)

# Index a tup
print(tup[0])
print(tup[-1])

# Tuples are ordered and unchangeable


## Sets
fruits = {'apple', 'mango', 'guava', 'banana', 'orange'}
print(fruits)

# Add to a set
fruits.add('watermelon')

# Sets cannot have duplicate values

# Sets are uordered

#Union

# Intersection

# Differennce

# Clear


# Object Oriented Programming
## Creating a class
class Car:
    # Class attribute
    category = "Vehicle"

    # Constructor method
    def __init__(self, brand, model):
        # Instance attributes
        self.brand = brand
        self.model = model

    # Method
    def display_info(self):
        print(f"Brand: {self.brand}, Model: {self.model}")

# Creating objects of the Car class
car1 = Car("Toyota", "Corolla")
car2 = Car("Honda", "Civic")

# Accessing class attribute
print(Car.category)

# Accessing instance attributes
car1.display_info()
car2.display_info()

# Attributes
car1.brand = "Ford" 
car2.year = 2020     

print(car1.brand)    
print(getattr(car2, "year", "N/A"))

# Modules
# Importing mymodule module
import mymodule

# Using function from the module
mymodule.greet("Alice")

# Creating object of MyClass from the module
obj = mymodule.MyClass(5)
print(obj.x)


## Main Principles of OOP
# Encapsulation
class Car:
    def __init__(self, make, model):
        self.make = make
        self.model = model

    def get_make(self):
        return self.make

    def get_model(self):
        return self.model

car = Car("Toyota", "Corolla")
print(car.get_make()) 
print(car.get_model())


# Polymorphism
class Animal:
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof"

class Cat(Animal):
    def make_sound(self):
        return "Meow"

def animal_sound(animal):
    return animal.make_sound()

dog = Dog()
print(animal_sound(dog))

cat = Cat()
print(animal_sound(cat))


# Modularity
import math_operations

print(math_operations.add(5, 3))     
print(math_operations.subtract(5, 3)) 
print(math_operations.multiply(5, 3))  
print(math_operations.divide(6, 3)) 


# Inheritance
class Animal:
    def speak(self):
        return "Animal speaks"

class Dog(Animal):
    def bark(self):
        return "Woof"

dog = Dog()
print(dog.speak()) 
print(dog.bark())  