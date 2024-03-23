# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



## Variables
print(5+5)
print(5+5+5)

score = 5
print(score)
score = score + 5
print(score)

new_score = score + 2
print(new_score)




## Data types
# Strings
name = "John"
print(name)
print(type(name))

# "" '' """ """ ''' '''

name = 'John'
print(name)

word = "I am \
a multiline string"

print(word)

word = """I 
am
also
a
multiline string"""
print(word)

# Integers
age = 25
print(age)
print(type(age))

# Floats
tax_percent = 0.1
print(tax_percent)
print(type(tax_percent))

# Booleans true/false
# test = true
is_student = True
print(is_student)
print(type(is_student))

# Complex
complex_num = 1 + 2j
print(complex_num)
print(type(complex_num))

# None
nothing = None
print(nothing)
print(type(nothing))

# List []
ages = [23, 34, 45]
print(type(ages))

# Tuple ()
ages = (23, 34, 45)
print(type(ages))

# Set {}
ages = {23, 34, 45}
print(type(ages))

# Dictionary {} key-value pairs
person = {'name': 'John', 'age': 25, 'email': 'i@gmail.com'}
print(type(person))




## Functions
# Function definition
def add_2_numbers(a, b):
    return a + b

# Function call
print(add_2_numbers(5, 5)) # arguments = 5, 5

# Empty function
def not_sure():
    pass

# Function that calls another function
# To get product of sums
## 1. Add the numbers together
## 2. Multiply the sums
def product_of_sums(a, b, c, d): # parameters = a, b, c, d
    return add_2_numbers(a, b) * add_2_numbers(c, d)

pos = product_of_sums(1, 2, 3, 4) # arguments = 1, 2, 3, 4
print(pos)

# Recursive function
#  Factorial = n * n-1 * n-2 * ... * 1
def factorial(num): # parameter = num
    if num == 0:
        return 1
    else:
        return num * factorial(num - 1)
# 4*3*2*1*1
    
print(factorial(10)) # argument = 10

# Lambda functions
add = lambda x, y: x + y
print(add(2, 3))

multiply = lambda a, b: a * b
print(multiply(2, 3))

# Local and Global variables
a = 10

def func():
    a = 5
    
print(a)

# Global
def func():
    global e
    e = 5

func()
print(e)

# Local
def func():
    c = 5

func()
# print(c)



## Control flow
# Conditional statements
a = 5
print("Is a equal to 5?")
print(a == 5) # Is a equal to 5?
print("Is a equal to 10?")
print(a == 10) # Is a equal to 10?
print("Is a not equal to 5?")
print(a != 5) # Is a not equal to 5?
print("Is a greater than 5?")
print(a > 5)
print("Is a less than 5?")
print(a < 5)
print("Is a greater than or equal to 5?")
print(a >= 5)
print("Is a less than or equal to 5?")
print(a <= 5)

# Equality: ==
# Inequality: !=
# Greater than: >
# Less than: <
# Greater than or equal to: >=
# Less than or equal to: <=

name = "John"
print(name == "john")
print(name == "Mary")

# Boolean Operators
# and- true if both are true
print(True and True)
print(True and False)
print(False and False and True)

# or - true if one is true
print(True or True)
print(True or False)
print(False or False)

# not - true if false and vice versa
print(not True)
print(not False)

# Intergers in logic
print(int(True))
print(int(False))

# 1 = True & 0 = False
# Every number greater or equal to 1 is True
print(bool(34))

# not 
print(not 0)
print(not 1)

# Converting from boolean to integer
int_ = int(False)
print(int_)

# Converting from integer to boolean
bool_ = bool(1)
print(bool_)

# Keywords in python: int, str, print, input, bool, float...

# Conditional + Boolean
a = 4
b = 7
c = 10

# 4<7 & 7<10
# True & True 
# True
print(a < b and b < c)

# if
a = 7
if a == 5:
    print("a is 5")

# if else
a = 10
if a == 5:
    print("a is 5")
else:
    print("a is not 5")

# if elif else
# elif is actually else-if
height = 4.5
if height < 5:
    print("Small")    
elif height < 6:
    print("Medium")
else:
    print("Large")

# Simple GPA Calculator
score = 30
if score >= 70:
    print("A")
elif score >= 60:
    print("B")
elif score >= 50: # case 50:
    print("C")
elif score >= 40:
    print("D")
else:
    print("F")

# Nested if
a = 6
b = 10
if a == 5:
    if b == 10:
        print("a is 5 and b is 10")
    else:
        print("a is 5 and b is not 10")
else:
    if b == 10:
        print("a is not 5 and b is 10")
    else:
        print("a is not 5 and b is not 10")

# Switch-Case
# Response codes
response_code = 201
match response_code:
    case 200: # case when response_code is 200: perform this
        print("OK")
    case 201:
        print("Created")
    case 300:
        print("Multiple choices")
    case 307:
        print("Temporary Redirect")
    case 404:
        print("Not found")
    case 500:
        print("Internal Server Error")
    case 502:
        print("Bad Gateway")

# while loop
# a loop is a piece of code that runs repeatedly until a certain condition is reached
# In python we have the while loop and for loop
# while loop - runs as long as the condition is true
# for loop - runs for a specific number of times
        
# Avoiding infinite loops:
    # 1. Use incremental or decremental statements
    # 2. Use the break statement
        
# Countdown
# Use a decremental statement when the variable is greater than the condition       
a = 4
while a != 0:
    print(a)
    a = a-1 # Decremental statement

# Use an incremental statement when the variable is less than the condition
b = 0
while b < 5:
    print(b)
    b = b + 1

# for loop - iterate through a sequence (list, tuple, dictionary, string)
names = ['John', 'James', 'Jack', 'Jeremy', 'Joseph']
for name in names:
    print(name)

for integer in [1, 4, 2, 6, 4]:
    print(integer)

word = "[AI] Academy Python!"
for letter in word:
    print(letter)

# for loop + range function
# range(start, stop, step)
# start - the value the array starts from; the first value in the array/list/sequence
# stop - 1 less the value the array stops at; the last value in the array/list/sequence
# step - the increment or decrement value

for num in range(0, 11):
    print(num)    

# print(range(1, 10))
    
# 0-1 = -1
# Simple countdown function in python
def countdown(num):
    for count in range(num, -1, -1):
        print(count)

countdown(30)


# break, continue & pass statements ## Alter the flow loops & functions
# break statement - to exit a loop
while True:
    footballer = input("Type in a footballer: ") # input accepts a value from the user & print outputs a value
    if footballer == 'Ronaldo':
        print(5)
        break


    
# continue statement - jump back to the start of a loop
for num in range(0,10):
    if num == 5:
        continue
    print(f'Iteration: {num}')

# pass statement - it does nothing & serves as a placeholder in a function
for num in range(0,10):
    if num == 5:
        pass
    print(f'Iteration: {num}')



