def display_student(name, age):
	print("Student Name: ", name)
	print("Age: ", age)

name = input("Enter Student Name: ")
age = input("Enter Age: ")

# Rename the function and call it with a new name
show_student = display_student

display_student(name, age)
