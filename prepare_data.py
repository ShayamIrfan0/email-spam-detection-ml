import pandas as pd
from pathlib import Path

# Location of the downloaded dataset
source_file = Path("SMSSpamCollection")

# Location where the CSV will be saved
output_file = Path("data/spam.csv")

# Check whether the dataset exists
if not source_file.exists():
    raise FileNotFoundError(
        "SMSSpamCollection file not found!"
    )

# Create the data folder if necessary
output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Read the dataset
df = pd.read_csv(
    source_file,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="utf-8"
)

# Save the dataset as CSV
df.to_csv(output_file, index=False)

print("Dataset converted successfully!")
print(df.head())

print("\nSpam and legitimate message counts:")
print(df["label"].value_counts())