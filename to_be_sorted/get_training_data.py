# This builds a dataset for training a linear probe to classify tokens according to various criteria
# criterion: should be "anywhere", "starting", "length" or "distinct"
# target: should be an all-Roman string if criterion == "anywhere" or "starting"
#  or else should be a positive integer in string form if criterion = "length" or "distinct"

def get_training_data(criterion, target, num_samples, embeddings, all_rom_token_gt2_indices):

	# Fetch indices for tokens that match the required pattern
	if criterion == "anywhere": #required pattern is target string appears anywhere as substring of token
		if not target.isalpha():
			return None
		else:		
			positive_indices = [index for index in all_rom_token_gt2_indices if target.lower() in token_strings[index].lstrip().lower()]

	elif criterion == "starting": #required pattern is target string appears as initial substring of lstrip'd token
		if not target.isalpha():
			return None
		else:		
			positive_indices = [index for index in all_rom_token_gt2_indices if token_strings[index].lstrip().lower().startswith(target.lower())]


	elif criterion == "length": #required pattern is token is of length int(target)
		if not target.isdigit() or (target.isdigit() and int(target) == 0):
			return None
		else:
			positive_indices = [index for index in all_rom_token_gt2_indices if len(token_strings[index].lstrip()) == int(target)]

	elif criterion == "distinct": #required pattern is token has int(target) distinct characters
		if not target.isdigit() or (target.isdigit() and int(target) == 0):
			return None
		else:
			positive_indices = [index for index in all_rom_token_gt2_indices if len(set(token_strings[index].lstrip())) == int(target)]

	# Fetch indices for tokens that do not begin this way
	# (by taking a set difference and then converting back to a list)
	negative_indices = list(set(all_rom_token_gt2_indices) - set(positive_indices))

	# Randomly sample from positive and negative indices to balance the dataset
	num_positive = min(num_samples // 2, len(positive_indices))
	num_negative = num_samples - num_positive

	sampled_positive_indices = random.sample(positive_indices, num_positive)
	sampled_negative_indices = random.sample(negative_indices, num_negative)

	# Combine sampled indices
	sampled_indices = sampled_positive_indices + sampled_negative_indices
	random.shuffle(sampled_indices)  # Shuffle combined indices for randomness in training

	# Extract corresponding embeddings and labels
	all_embeddings = embeddings[sampled_indices]
	all_labels = [1 if idx in positive_indices else 0 for idx in sampled_indices]

	return all_embeddings.clone().detach(), torch.tensor(all_labels).clone().detach()