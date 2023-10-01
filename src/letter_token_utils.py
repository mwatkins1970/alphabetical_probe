ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def get_token_strings(tokenizer):
    '''
    THIS MAKES A LIST OF THE 50257 SIGNIFICANT TOKENS, AS WELL AS  SUB-LISTS OF ALL-ROMAN TOKENS
    '''
    token_strings = [tokenizer.decode([i]) for i in range(50257)]
    return token_strings

def get_all_rom_tokens(token_strings):
    '''
    The leading spaces in these token strings are weird, they somehow delete themselves and the character before whatever they get appended to
    ...so you got stuff like "The string'petertodd' spelled in all capital letters..."
    ...rather than "The string ' petertodd' spelled is..." as it should be
    This seems to be an easy fix. Just go through the lists and replace the first character with an actual space!
    '''
    num_tokens = len(token_strings)
    print("There are " + str(num_tokens) + " tokens.")
    
    # replace the first character with an actual space!
    for token in token_strings:
        token = " " + token[1:]

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

    # all_rom_token_gt2_indices = [idx for idx in all_rom_token_indices if len(token_strings[idx].lstrip()) > 2]

    print(f"There are {len(all_rom_tokens)} all-Roman tokens.")

    return all_rom_tokens, all_rom_token_indices, 

'''
Build dictionaries of tokens for 'contains <letter>' and 'starts with <letter>' with one key per letter of the alphabet also build dictionaries of tokens for 'has total of <num_letters> letter' and 'has <num_letters> distinct letters.
'''


def get_letter_presence_dict(all_rom_token_indices, token_strings):
    letter_presence_dict = {}

    for letter in ALPHABET:
        letter_presence_dict[letter] = []

        for j in range(len(all_rom_token_indices)):
            if letter in token_strings[all_rom_token_indices[j]].upper():
                letter_presence_dict[letter].append(all_rom_token_indices[j])

    print(f"There are {len(letter_presence_dict[letter])} all-Roman tokens containing {letter} or {letter.lower()}")

    return letter_presence_dict

def get_letter_starting_dict(all_rom_token_indices, token_strings):

    letter_starting_dict = {}

    for letter in ALPHABET:
        letter_starting_dict[letter] = []

        for j in range(len(all_rom_token_indices)):
            if letter == token_strings[all_rom_token_indices[j]].lstrip()[0].upper():
                letter_starting_dict[letter].append(all_rom_token_indices[j])

        print(f"There are {len(letter_starting_dict[letter])} all-Roman tokens starting {letter} or {letter.lower()}")

    return letter_starting_dict

def get_token_length_dict(all_rom_token_indices, token_strings):

    token_length_dict = {}
    for num_letters in range(1,15):

        token_length_dict[num_letters] = []

        for j in range(len(all_rom_token_indices)):
            if len(token_strings[all_rom_token_indices[j]].lstrip()) == num_letters:
                token_length_dict[num_letters].append(all_rom_token_indices[j])

    print(f"There are {len(token_length_dict[num_letters])} all-Roman tokens with {num_letters} letters")

    return token_length_dict

def get_distinct_letters_dict(all_rom_token_indices, token_strings):
  
    distinct_letters_dict = {}
    for num_distinct_letters in range(1,15):

        distinct_letters_dict[num_distinct_letters] = []

        for k in range(len(all_rom_token_indices)):
            
            if len(set([char for char in token_strings[all_rom_token_indices[k]].lstrip().lower() if char.isalpha()])) == num_distinct_letters:
                distinct_letters_dict[num_distinct_letters].append(all_rom_token_indices[k])

    print(f"There are {len(distinct_letters_dict[num_distinct_letters])} all-Roman tokens with {num_distinct_letters} distinct letters")

    return distinct_letters_dict