from transformers import AutoTokenizar, AutoModelForMaskedLM
import pandas as pd
import numpy as np
from scipy.special import softmax

model_name = 'bert-base-cased'

tokenizer = AutoTokenizar.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

mask = tokenizer.mask_token

sentence = f'I want to {mask} pizza for tonight'

tokens = tokenizer.tokenize(sentence)

# Encoding the input sentence and getting model predictions  --> Model Prediction
encoded_inputs = tokenizer(sentence, return_tensors="pt")
output = model(**encoded_inputs)

# Detaching the logits from the model output and converting to numpy array
logits = output.logits.detach().numpy()[0]

# Extracting the logits for the masked token and calculating the confidence scores  --> Analyzing predictions
masked_logits = logits[tokens.index(mask) + 1]
confidence_scores = softmax(masked_logits)

# Displaying top predictions
# Iterating over the top 5 predicted tokens and printing the sentences with the masked token replaced
for i in np.argsort(confidence_scores)[::-1][:5]:
    pred_token = tokenizer.decode(i)
    score = confidence_scores[i]

    # print(pred_token, score)
    print(sentence.replace(mask, pred_token))
    
    



# Requirements:
#  !pip install -U transformers
