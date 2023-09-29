# THIS MAKES A LIST OF THE 50257 SIGNIFICANT TOKENS, AS WELL AS  SUB-LISTS OF ALL-ROMAN TOKENS

# Define the file path for the token_strings list
file_path = os.path.join(GDRIVE_PATH, "token_strings.pkl")

if os.path.exists(file_path):     # if it's already saved, load it
  with open(file_path, 'rb') as file:
      token_strings = pickle.load(file)
else:                             # otherwise create it and sav it.
  token_strings = [tokenizer.decode([i]) for i in range(50257)]
  with open(file_path, 'wb') as file:
      pickle.dump(token_strings, file)

num_tokens = len(token_strings)

# The leading spaces in these token strings are weird, they somehow delete themselves and the character before whatever they get appended to
#...so you got stuff like "The string'petertodd' spelled in all capital letters..."
#...rather than "The string ' petertodd' spelled is..." as it should be
# This seems to be an easy fix. Just go through the lists and replace the first character with an actual space!
for token in token_strings:
  token = " " + token[1:]

print("There are " + str(num_tokens) + " tokens.")

all_rom_tokens = []			#initialise list of all-roman tokens
all_rom_token_indices = []

for i in range(num_tokens):
	all_rom = True                       # Is token_string[i] all roman characters? Assume to begin that it is.
	for ch in range(len(token_strings[i])):
		if token_strings[i][ch] not in ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
			all_rom = False
	if all_rom == True and token_strings[i] not in [' ', '  ', '   ', '    ', '     ', '      ', '       ', '        ']:
		all_rom_tokens.append(token_strings[i])
		all_rom_token_indices.append(i)

all_rom_token_gt2_indices = [idx for idx in all_rom_token_indices if len(token_strings[idx].lstrip()) > 2]


print(f"There are {len(all_rom_tokens)} all-Roman tokens.")
print(f"There are {len(all_rom_token_gt2_indices)} all-Roman tokens with more than two letters.")