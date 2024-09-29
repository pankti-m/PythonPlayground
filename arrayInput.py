# Initialize an array of size 3
names = [None] * 3

for i in range(0, 3):
    name_str = "Enter name" + str(i+1) + ":" 
    names[i] = input(name_str)

print(names)
