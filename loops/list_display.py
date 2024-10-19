# Iterate through a list and display numbers according to the following condition:
# The number must be divisible by five
# If the number is greater than 150, then skip it and move to the following number
# If the number is greater than 500, then stop the loop

input_list = [12, 75, 150, 180, 145, 525, 50]

for num in input_list:
    if (num > 500):
        break
    if (num > 150):
        continue
    if num%5 == 0:
        print(num)



