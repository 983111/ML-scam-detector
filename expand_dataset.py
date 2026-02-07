import csv, random
from feature_extractor import extract_features

scam_texts = [
    "URGENT verify your account",
    "Free job offer earn money",
    "Suspicious login detected",
    "You won a lottery prize",
    "Limited time investment"
]

safe_texts = [
    "Please review the documentation",
    "Meeting scheduled tomorrow",
    "Project update attached",
    "Watch tutorials online",
    "Monthly report available"
]

def mutate(text):
    noise = [" now", " please", " today", " asap", "!!!", ""]
    return text + random.choice(noise)

rows = []

for _ in range(600):
    t = mutate(random.choice(scam_texts))
    rows.append(extract_features(t, random.randint(30, 90)) + [1])

for _ in range(400):
    t = mutate(random.choice(safe_texts))
    rows.append(extract_features(t, random.randint(-10, 10)) + [0])

random.shuffle(rows)

with open("scam_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([f"f{i}" for i in range(1, 17)] + ["label"])
    writer.writerows(rows)

print("✅ Production dataset created")
