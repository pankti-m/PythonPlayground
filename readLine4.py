# Prints 4th line in a file

file_name = input("Enter file name: ")

with open(file_name, 'r') as fp:
    for i in range(1, 5):
        line = fp.readline()
        if line == '':
            print("File has less than 4 lines.")
            break
        if i == 4:
            print("Printing 4th line.")
            print(line)


    
