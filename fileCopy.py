# Copies all lines from a file to another file except line 5

file_name = input("Enter path of the file: ")

with open(file_name, 'r') as fp1:
    with open('outputFile.txt', 'x') as fp2:
        i = 0;
        line = fp1.readline()
        while(line != ''):
            i += 1
            if i != 5 :
                print("Writing", line)
                fp2.write(line)
            line = fp1.readline()
