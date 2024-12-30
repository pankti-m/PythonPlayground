num = 76542
reverse = 0

while num > 0:
    remainder = num%10
    num = num//10
    reverse = reverse*10 + remainder
print(reverse)
