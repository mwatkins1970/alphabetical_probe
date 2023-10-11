# An approach to demonstrating GPT-J's ability to answer binary questions about letter presence
# It iterates through all all-Roman tokens with > 2 characters, and with 0.5 probability randomly chooses a
# letter either in or not in the token string. A multishot prompt elicits a list of letters supposedly in the token
# Scores are calculated according to whether the output rightly/wrongly 'thinks' the target letter is in this list

# this still needs wandb to be included

import random
import re
device = "cpu"

alphabet = set("abcdefghijklmnopqrstuvwxyz")

TP = 0
TN = 0
FP = 0
FN = 0

accuracy = 0
f1_score = 0

for idx in all_rom_token_gt2_indices:

    # Pick a random token from all_rom_tokens
    token = random.choice(all_rom_tokens)
    unique_chars = set(token.lower().lstrip())
    remaining_chars = alphabet - unique_chars

    # Pick a random boolean
    boolean_choice = random.choice([True, False])

    if boolean_choice:
        target_letter = random.choice(list(unique_chars))
    else:
        target_letter = random.choice(list(remaining_chars))

    print(f"TOKEN: {token}")
    print(f"TARGET LETTER: {target_letter}")

    letter_present = target_letter in token.lower()

    print(f"LETTER PRESENT?: {letter_present}")

    prompt = f'''The string " heaven" contains the letters "H", "E", "A", "V" and "N"\nThe string " Kardashian" contains the letters "K", "A", "R", "D", "S", "H", "I" and "N"\nThe string "box" contains the letters "B", "O" and "X"\nThe string "{token}" contains the letters "'''

    print(f"PROMPT:\n{prompt}")

    ix = tokenizer.encode(prompt)

    model_out = GPTmodel.generate(
            torch.tensor(ix).unsqueeze(0).to(device),
            max_length=len(ix) + 40,
            temperature=0.00000000001,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
    )

    output = tokenizer.decode(model_out[0])[len(prompt):]  
    output = output.lstrip().split('\n')[0] 
    print(output) 

    matches = re.findall(r'(?<=")[A-Z](?=")', '"' + output)
    matches = list(set(matches))
    matches = [letter.lower() for letter in matches]
    print(matches)
  
    if target_letter in matches and boolean_choice == True:
        TP += 1
    elif target_letter in matches and boolean_choice == False:
        FP += 1
    elif target_letter not in matches and boolean_choice == True:
        FN += 1
    elif target_letter not in matches and boolean_choice == False:
        TN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Ensure we don't divide by zero
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    # Ensure we don't divide by zero
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"TP: {TP}; TN: {TN}; FP: {FP}; FN: {FN}")
    print(f"ACCURACY: {accuracy}")
    print(f"F1 SCORE: {f1_score}")

    print('\n')

