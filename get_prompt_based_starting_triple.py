def get_prompt_based_starting_triple(idx):

  token = token_strings[idx]
  print("Prompt-based prediction:")

  prompt1 = f'The string "house", spelled in all capital letters separated by hyphens, looks like this: H-O-U-S-E\nThe string " Through", spelled in all capital letters separated by hyphens, looks like this: T-H-R-O-U-G-H\nThe string "{token}", spelled in all capital letters separated by hyphens, looks like this: '

  ix1 = tokenizer.encode(prompt1)

  model_out = GPTmodel.generate(
      torch.tensor(ix1).unsqueeze(0).to(device),
      max_length=len(ix1) + 1,
      temperature=0.00000000001,
      do_sample=True,
      pad_token_id=tokenizer.eos_token_id,
  )

  output = tokenizer.decode(model_out[0])

  pred_first_letter = output[-1].upper()

  print(f"PREDICTED FIRST LETTER: {pred_first_letter}")

  prompt2 = f'The string "house", spelled in all capital letters separated by hyphens, looks like this: H-O-U-S-E\nThe string " Through", spelled in all capital letters separated by hyphens, looks like this: T-H-R-O-U-G-H\nThe string "{token}", spelled in all capital letters separated by hyphens, looks like this: {pred_first_letter.upper()}-'

  ix2 = tokenizer.encode(prompt2)

  model_out = GPTmodel.generate(
      torch.tensor(ix2).unsqueeze(0).to(device),
      max_length= len(ix2) + 1,
      temperature=0.00000000001,
      do_sample=True,
      pad_token_id=tokenizer.eos_token_id,
  )

  output = tokenizer.decode(model_out[0])

  pred_second_letter = output[-1].upper()

  print(f"PREDICTED SECOND LETTER: {pred_second_letter}")

  prompt3 = f'The string "house", spelled in all capital letters separated by hyphens, looks like this: H-O-U-S-E\nThe string " Through", spelled in all capital letters separated by hyphens, looks like this: T-H-R-O-U-G-H\nThe string "{token}", spelled in all capital letters separated by hyphens, looks like this: {pred_first_letter.upper()}-{pred_second_letter.upper()}-'

  ix3 = tokenizer.encode(prompt3)

  model_out = GPTmodel.generate(
      torch.tensor(ix3).unsqueeze(0).to(device),
      max_length= len(ix3) + 1,
      temperature=0.00000000001,
      do_sample=True,
      pad_token_id=tokenizer.eos_token_id,
  )

  output = tokenizer.decode(model_out[0])

  pred_third_letter = output[-1].upper()

  print(f"PREDICTED THIRD LETTER: {pred_third_letter}\n")

  return pred_first_letter + pred_second_letter + pred_third_letter