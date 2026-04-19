def print_dictionary(**kwargs):
	for key, value in kwargs.items():
		print(f"{key}: {value}", end=" ")
	print()

print_dictionary(fruit="apple", color="yellow", taste="sweet")
print_dictionary(fruit="kiwi", color="green", taste="sour")
