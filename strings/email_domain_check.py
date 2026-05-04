def filter_valid_senders(senders):
	list = []
	for sender in senders:
		if (sender.endswith("@broadcom.com")):
			list.append(sender)
	return ", ".join(list)

senders = ["johndoe@gmail.com", "pankti.majmudar@gmail.com", "pankti.majmudar@broadcom.com", "xyz@abc.com", "xyz@Broadcom.com"]

all_senders = ", ".join(senders)
all_senders_str = f"All Senders: {all_senders}"
print(all_senders_str)

filtered_senders = filter_valid_senders(senders)
filtered_senders_str = f"Out of all senders, the allowed senders based on @broadcom.com domain are: {filtered_senders}."
print(filtered_senders_str)
