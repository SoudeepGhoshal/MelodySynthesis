import json
from sklearn.model_selection import train_test_split

# Load your melody dataset
with open('processed_data/dataset.json', 'r') as file:
    melodies = json.load(file)

# 2. Perform the splits
# First split (train + temp)
train_size = 0.8  # 80% for training
melodies_train, melodies_temp = train_test_split(
    melodies, test_size=(1 - train_size), random_state=42, shuffle=True
)

# Second split (test + validation) from the temporary set
test_size = 0.14 / (1 - train_size)
melodies_test, melodies_val = train_test_split(
    melodies_temp, test_size=(1 - test_size), random_state=42, shuffle=True
)

# Save splits into separate files
with open('processed_data/train.json', 'w') as file:
    json.dump(melodies_train, file, indent=2)

with open('processed_data/validation.json', 'w') as file:
    json.dump(melodies_val, file, indent=2)

with open('processed_data/test.json', 'w') as file:
    json.dump(melodies_test, file, indent=2)

print(f"Dataset split completed successfully:")
print(f"Training set size: {len(melodies_train)}")
print(f"Validation set size: {len(melodies_val)}")
print(f"Testing set size: {len(melodies_test)}")
