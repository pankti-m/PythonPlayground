alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

index = int(input("Enter the Alphabet Index: ")) - 1

if (index >= 0 and index < 26):
	to_print = f"Alphabet at the given index is {alphabets[index]}"
	print(to_print)
else:
	print("")
