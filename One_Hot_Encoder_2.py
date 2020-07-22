import pandas as pd
import numpy as np

statement = 'thanksforcomingtoday'
alphabet = 'abcdefghijklmnopqrstuvwxyz'
int_char_map = [(i,count) for i,count in enumerate(alphabet)]
df = pd.DataFrame(int_char_map)

char_to_int = dict((c,i) for i,c in enumerate(alphabet))
int_to_char = dict((i,c) for i,c in enumerate(alphabet))
integer_encoded = [char_to_int[char] for char in statement]

one_hot_encoded = list()
for value in integer_encoded:
    letter = [0 for _ in range(len(alphabet))]
    letter[value] = 1
    one_hot_encoded.append(letter)
print(one_hot_encoded)
