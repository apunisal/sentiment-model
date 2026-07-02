import torch
import torch.nn as nn
import torch.optim as optim
import hashlib

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1) 
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)

VOCAB_SIZE = 15000  # Increased vocab size to accommodate the larger lexicon
EMBED_DIM = 64
HIDDEN_DIM = 32
SEQ_LEN = 20

model = SentimentModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)

def hash_token(token):
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % VOCAB_SIZE

def encode_text(text):
    words = text.lower().split()
    tokens = []
    i = 0
    while i < len(words):
        if words[i] in ["not", "no", "never", "dont", "cant", "isnt", "wasnt"] and (i + 1) < len(words):
            combined_token = f"{words[i]}_{words[i+1]}"
            tokens.append(hash_token(combined_token))
            i += 2
        else:
            tokens.append(hash_token(words[i]))
            i += 1
            
    if len(tokens) > SEQ_LEN:
        return tokens[:SEQ_LEN]
    else:
        return tokens + [0] * (SEQ_LEN - len(tokens))

print("Compiling advanced high-vocabulary dataset...")

# --- CATEGORIZED HIGH-VOCABULARY DICTIONARY ---

# Category 1: Standard Positive Base
base_pos = ["love", "like", "enjoy", "happy", "great", "awesome", "excellent", "wonderful", "beautiful", "good", "best", "excited"]

# Category 2: Premium & Executive Praise
premium_pos = ["exceptional", "outstanding", "impeccable", "exemplary", "unparalleled", "unrivaled", "distinguished", "magnificent", "superb"]

# Category 3: Joy & Deep Satisfaction
joy_pos = ["ecstatic", "elated", "jubilant", "delighted", "rapturous", "euphoric", "content", "satisfied", "blissful", "joyous"]

# Category 4: Innovation & Success
success_pos = ["brilliant", "ingenious", "groundbreaking", "revolutionary", "phenomenal", "triumphant", "successful", "prosperous", "flourishing"]

# Category 5: Standard Negative Base
base_neg = ["hate", "dislike", "angry", "frustrated", "terrible", "bad", "sad", "awful", "horrible", "useless", "broken", "failed"]

# Category 6: Severe Disappointment & Chaos
severe_neg = ["catastrophic", "disastrous", "abysmal", "atrocious", "deplorable", "execrable", "appalling", "dreadful", "reprehensible"]

# Category 7: Sadness & Exhaustion
exhaust_neg = ["miserable", "melancholy", "despondent", "disconsolate", "gloomy", "fatigued", "exhausted", "lethargic", "drained", "weary"]

# Category 8: Defective & Malfunctioning
defect_neg = ["faulty", "deficient", "substandard", "detrimental", "flawed", "erroneous", "obsolete", "dysfunctional", "inadequate"]

# Combine into master structural lists
all_pos_words = base_pos + premium_pos + joy_pos + success_pos
all_neg_words = base_neg + severe_neg + exhaust_neg + defect_neg

starters = [
    "it is", "this is", "i feel", "everything looks", "highly", "absolutely", "truly", 
    "my day was", "the service was", "the food tasted", "my experience was", "the weather is"
]

training_data = []

# Manual context anchors
training_data.append(("i absolutely love this new phone the battery life is amazing", 1.0))
training_data.append(("the service was terrible and my food was cold", 0.0))
training_data.append(("i lost my key", 0.0))
training_data.append(("its raining and i dont have raincoat", 0.0))

# 1. Generate multi-word contextual combinations
for s in starters:
    for p in all_pos_words:
        training_data.append((f"{s} {p}", 1.0))
    for n in all_neg_words:
        training_data.append((f"{s} {n}", 0.0))

# 2. Generate explicit Negation Structures (masters "not bad" vs "not good")
negators = ["not", "no", "never", "dont", "cant", "isnt", "wasnt"]
for s in starters:
    for neg in negators:
        for p in all_pos_words:
            training_data.append((f"{s} {neg} {p}", 0.0))  # "not impeccable" -> Negative
        for n in all_neg_words:
            training_data.append((f"{s} {neg} {n}", 1.0))  # "not abysmal" -> Positive

inputs = torch.tensor([encode_text(text) for text, label in training_data])
labels = torch.tensor([label for text, label in training_data]).unsqueeze(1)

print(f"Dataset compiled! Total training patterns: {len(training_data)}")
print("Training structural neural network...")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(40): 
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/40 - Loss: {loss.item():.4f}")

print("Exporting model to ONNX...")
model.eval()
onnx_dummy_input = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
torch.onnx.export(
    model, onnx_dummy_input, "sentiment_model.onnx", 
    export_params=True, opset_version=18,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size', 1: 'seq_len'}, 'output': {0: 'batch_size'}}
)
print("High-vocabulary model exported cleanly!")
