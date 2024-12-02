num = int(input("Enter a number: "))
copy = num
digits = 0;
while (copy > 0):
    copy = copy/10;
    digits += 1
print("Number of digits in " +  str(num) + " is " + str(digits))
