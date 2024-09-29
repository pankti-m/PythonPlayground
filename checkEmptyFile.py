# Checks if a file is empty

import os

file_name = input("Enter file name: ")

if (os.stat(file_name).st_size == 0):
    print("File ", file_name, " is empty")
else:
    print("File ", file_name, " is not empty")
