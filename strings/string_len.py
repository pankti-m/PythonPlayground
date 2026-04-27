full_name = input("Enter your First and Last Name, separated by space: ")
name_len = len(full_name.replace(" ", ""))
f_string = f"Hello {full_name}.  Your full name has {name_len} characters excluding spaces"
print(f_string)
