from transformers import BertModel, AutoTokenizer
from scipy.spatial.distance import cosine

# Defining the model name
model_name = "bert-base-cased"

# Loading the pre-trained model and tokenizer
model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Defining a function to encode the input text and get model predictions
def predict(text):
    encoded_inputs = tokenizer(text, return_tensors="pt")
    return model(**encoded_inputs)[0]

# 📃 Defining the Sentences

# Defining the sentences
sentence1 = "There was a fly drinking from my soup"
sentence2 = "There is a fly swimming in my juice"
# sentence2 = "To become a commercial pilot, he had to fly for 1500 hours." # second fly example

# Tokenizing the sentences
tokens1 = tokenizer.tokenize(sentence1)
tokens2 = tokenizer.tokenize(sentence2)


# 🔍 Tokenization and Model Predictions

# Getting model predictions for the sentences
out1 = predict(sentence1)
out2 = predict(sentence2)

# Extracting embeddings for the word 'fly' in both sentences
emb1 = out1[0:, tokens1.index("fly"), :].detach()[0]
emb2 = out2[0:, tokens2.index("fly"), :].detach()[0]

# Calculating the cosine similarity between the embeddings
print(cosine(emb1, emb2))

