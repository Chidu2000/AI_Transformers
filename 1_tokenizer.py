from transformers import BertModel, AutoTokenizer
import pandas as pd

model_name = 'bert-base-cased'

model = BertModel.from_pretrained(model_name) # collect weights from the pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_name) # tokenize the model

sentence = "When life gives you lemons, don't make lemonade"

tokens = tokenizer.tokenize(sentence)

# print(tokens)

vocab = tokenizer.vocab
vocab_db = pd.DataFrame({"token": vocab.keys(), "token_id": vocab.values()})

vocab_df = vocab_df.sort_values(by='token_id').set_index('token_id')

token_ids = tokenizer.encode(sentence)

# Print the length of tokens and token_ids
print("Number of tokens:", len(tokens))
print("Number of token IDs:", len(token_ids))

# Access the tokens in the vocabulary DataFrame by index
print("Token at position 101:", vocab_df.iloc[101])
print("Token at position 102:", vocab_df.iloc[102])

# Zip tokens and token_ids (excluding the first and last token_ids for [CLS] and [SEP])
list(zip(tokens, token_ids[1:-1]))

# Decode the token_ids (excluding the first and last token_ids for [CLS] and [SEP]) back into the original sentence
tokenizer.decode(token_ids[1:-1])

# Tokenize the sentence using the tokenizer's `__call__` method
tokenizer_out = tokenizer(sentence)
print(tokenizer_out)

# Handling multiple sentences
# Create a new sentence by removing "don't " from the original sentence
sentence2 = sentence.replace("don't ", "")
print(sentence2)

# Tokenize both sentences with padding
tokenizer_out2 = tokenizer([sentence, sentence2], padding=True)
print(tokenizer_out2)

# Decode the tokenized input_ids for both sentences
print(tokenizer.decode(tokenizer_out2["input_ids"][0]))

print(tokenizer.decode(tokenizer_out2["input_ids"][1]))





