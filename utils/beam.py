import numpy as np

def beam_search_decoder(predictions, top_k=3):
    # start with an empty sequence with zero score
    # predictions =
    output_sequences = [([], 0)]

    # looping through all the predictions
    for token_probs in predictions:
        new_sequences = []

        # append new tokens to old sequences and re-score
        for old_seq, old_score in output_sequences:
            for char_index in range(len(token_probs)):
                new_seq = old_seq + [char_index]
                # print (token_probs[char_index])
                # considering log-likelihood for scoring
                # new_score = old_score + np.log(token_probs[char_index])
                new_score = old_score + np.log(token_probs[char_index])
                new_sequences.append((new_seq, new_score))

        # sort all new sequences in the de-creasing order of their score
        output_sequences = sorted(new_sequences, key=lambda val: val[1], reverse=True)

        # select top-k based on score
        # *Note- best sequence is with the highest score
        output_sequences = output_sequences[:top_k]

    return output_sequences