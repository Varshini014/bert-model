import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from intent_data import TRAIN_DATA

# Label mapping
LABEL2ID = {
    "SEARCH_BUSINESS": 0,
    "OUT_OF_SCOPE": 1,
    "CREATE_WHISTLE": 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Custom Dataset
class IntentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(LABEL2ID[item["label"]])
        }

# Load tokenizer & model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(LABEL2ID)
)

# Dataset & DataLoader
dataset = IntentDataset(TRAIN_DATA, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
model.train()
for epoch in range(3):
    print(f"\nEpoch {epoch+1}")
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print("Loss:", total_loss)

# Save model
model.save_pretrained("intent_bert_model")
tokenizer.save_pretrained("intent_bert_model")

print("\nTraining complete. Model saved.")
