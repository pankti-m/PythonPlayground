end = int(input("Enter a number: "))
total = 0

for i in range(1, end+1):
    total += i
print("Sum of numbers from 1 to " +  str(end) + " is " + str(total))
