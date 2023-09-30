# Build dictionaries of tokens for 'contains <letter>' and 'starts with <letter>' with one key per letter of the alphabet
# ...also build dictionaries of tokens for 'has total of <num_letters> letter' and 'has <num_letters> distinct letters'

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

letter_presence_dict = {}
letter_starting_dict = {}

for letter in alphabet:

  letter_starting_dict[letter] = []
  letter_presence_dict[letter] = []

  for j in range(len(all_rom_token_indices)):
    if letter in token_strings[all_rom_token_indices[j]].upper():
      letter_presence_dict[letter].append(all_rom_token_indices[j])
    if letter == token_strings[all_rom_token_indices[j]].lstrip()[0].upper():
      letter_starting_dict[letter].append(all_rom_token_indices[j])

  print(f"There are {len(letter_starting_dict[letter])} all-Roman tokens starting {letter} or {letter.lower()}")
  print(f"There are {len(letter_presence_dict[letter])} all-Roman tokens containing {letter} or {letter.lower()}")


token_length_dict = {}
distinct_letters_dict = {}

for num_letters in range(1,15):

  token_length_dict[num_letters] = []
  distinct_letters_dict[num_distinct_letters] = []

  for j in range(len(all_rom_token_indices)):
    if len(token_strings[all_rom_token_indices[j]].lstrip()) == num_letters:
      token_length_dict[num_letters].append(all_rom_token_indices[j])

  for k in range(len(all_rom_token_indices)):
    if len(set([char for char in token_strings[all_rom_token_indices[k]].lstrip().lower() if char.isalpha()])) == num_distinct_letters:
      distinct_letters_dict[num_distinct_letters].append(all_rom_token_indices[k])

  print(f"There are {len(token_length_dict[num_letters])} all-Roman tokens with {num_letters} letters")
  print(f"There are {len(distinct_letters_dict[num_distinct_letters])} all-Roman tokens with {num_distinct_letters} distinct letters")
