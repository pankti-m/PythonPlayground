def sumTo10(num):
    if num == 1:
        return 1
    return num + sumTo10(num-1)

total = sumTo10(10)
print("The sum of numbers from 1 to 10 is: ", total)
