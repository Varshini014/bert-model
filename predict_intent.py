import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Label mapping
ID2LABEL = {0: "SEARCH_BUSINESS", 1: "OUT_OF_SCOPE", 2: "CREATE_WHISTLE"}

# Load trained model
tokenizer = BertTokenizer.from_pretrained("intent_bert_model")
model = BertForSequenceClassification.from_pretrained("intent_bert_model")
model.eval()

# Example sentences to test
test_sentences = [
    "Find plumbers near me",
    "What's the weather today?",
    "I want to create a whistle for tutors"
]

# Predict
for sentence in test_sentences:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    print(f"Text: {sentence}")
    print(f"BERT Intent: {ID2LABEL[pred]}\n")
