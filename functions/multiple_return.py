def add_and_subtract(*args):
    addition = 0
    subtraction = 0
    init = True
    for num in args:
        addition = addition + int(num)
        if init == True:
            subtraction = int(num)
            init = False
        else:
            subtraction = subtraction - int(num)
    return (addition, subtraction)

# .split() creates a tuple of the arguments (without it, it is a single string)
numbers = input("Input some numbers separated by space: ").split()

# * unpacks each item from the tuple and passes it as multiple arguments to the function.  Without it, a single tuple is passed as argument
print("Sum and Difference between all input numbers is respectively : ", add_and_subtract(*numbers))
