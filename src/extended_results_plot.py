# This produces various plots from the extended_results dictionaries.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extended_results_plot_runner(extended_results):

    def ordinal(n):
        """Return ordinal string for an integer."""
        suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
        # I'm checking for 10-20 because those are the digits that
        # don't follow the normal counting scheme.
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            # the second parameter is a default.
            suffix = suffixes.get(n % 10, 'th')
        return f"{n}{suffix}"

    def compute_switch_values(row):
        token = row['Token']
        stripped_token = token.lstrip().upper()
        first_letter = stripped_token[0]

        for coeff, value in row.items():
            if isinstance(coeff, int):
                # The value is now expected to be a tuple: (letter, coefficient)
                # Unpack the tuple and check the letter part against the first_letter
                letter, _ = value  # Ignore the coefficient value
                if letter != first_letter:
                    switch_coeff = coeff
                    switch_letter = letter  # We're no longer taking the first character of a string, but the letter directly
                    try:
                        switch_pos = stripped_token.index(switch_letter) + 1
                    except ValueError:
                        switch_pos = -1
                    return switch_coeff, switch_letter, switch_pos

        # If no switch is found, return default values
        return -1, '', -1


    # Build data_dict from the above 'extended_results' dictionary
    data_dict = {}
    for prediction_dict in extended_results['predictions']:
        data_dict[prediction_dict['token']] = prediction_dict['mutation predictions']

    # Create the DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()
    df = df.rename(columns={"index": "Token"})

    # Apply the compute_switch_values function and split the returned tuple into three new columns
    df['Switch Coeff'], df['Switch Letter'], df['Switch Pos'] = zip(*df.apply(compute_switch_values, axis=1))

    print(df)

    total_tokens = len(df)
    print(f"TOTAL TOKENS: {total_tokens}")


    # Visualization

    # Histogram of the coefficient where prediction goes wrong
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Switch Coeff'], kde=True, bins=range(1,22,2), discrete=True)  # Adjusted kde and bins
    plt.xticks(range(2,21,2))  # Setting ticks for even numbers between 2 to 20
    plt.title('Histogram of coefficients where prediction first ceases to predict first letter')
    plt.show()


    positions = [2, 3, 4]

    # Define your custom, irregular set of x-values
    x_values = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    # When plotting the histogram, use these x-values as bins (edges), and also for setting x-ticks
    for n in positions:
        # Filtering the DataFrame based on the switch letter position
        filtered_df = df[df.apply(lambda row: len(row['Token']) > n and row['Token'][n].upper() == row['Switch Letter'], axis=1)]

        # Plotting the histogram for the filtered DataFrame
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_df['Switch Coeff'], kde=True, bins=x_values, discrete=True)  # Use the custom x-values as bins
        plt.xticks(x_values)  # Set x-ticks to match the custom x-values
        plt.title(f'Histogram of the coefficient where prediction goes wrong and predicts letter in position {n}')
        plt.show()


    # Count of how often it switches to the 2nd, 3rd letter, etc.
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df['Switch Pos'])
    plt.title('Switch letter position counts (all switch coeff values)')
    plt.show()

    # Print statistics
    positions = list(range(2, 9))  # From 2nd to 8th letter
    for pos in positions:
        switch_count = len(df[df['Switch Pos'] == pos])
        percentage = (switch_count / total_tokens) * 100
        print(f"Switched to {ordinal(pos)} letter: {switch_count} times ({percentage:.2f}%)")

    # Handle the special case for switch outside of token
    switch_outside_token = len(df[df['Switch Pos'] == -1])
    percentage_outside = (switch_outside_token / total_tokens) * 100
    print(f"Switched to letter outside token: {switch_outside_token} times ({percentage_outside:.2f}%)")
    print('-'*100 + '\n')  # A separator for better clarity in output

    unique_coeffs = sorted(df['Switch Coeff'].dropna().unique())

    for coeff in unique_coeffs:
        if coeff > 0:
            # Filter the dataframe for the current 'switch coeff' value
            filtered_df = df[df['Switch Coeff'] == coeff]
            total_for_coeff = len(filtered_df)

            # Visualize 'Switch Position' for the filtered dataframe
            plt.figure(figsize=(10, 6))
            sns.countplot(x=filtered_df['Switch Pos'])
            plt.title(f'Switch Letter Position Counts for switch coeff = {coeff}')
            plt.show()

            # Print statistics for this filtered dataframe
            positions = list(range(2, 9))  # From 2nd to 8th letter
            for pos in positions:
                switch_count = len(filtered_df[filtered_df['Switch Pos'] == pos])
                percentage = (switch_count / total_for_coeff) * 100
                print(f"For switch coeff={coeff}, Switched to {ordinal(pos)} letter: {switch_count} times ({percentage:.2f}%)")

            switch_outside_token = len(filtered_df[filtered_df['Switch Pos'] == -1])
            percentage_outside = (switch_outside_token / total_for_coeff) * 100
            print(f"For switch coeff={coeff}, Switched to letter outside token: {switch_outside_token} times ({percentage_outside:.2f}%)")
            print('-'*100 + '\n')  # A separator for better clarity in output
