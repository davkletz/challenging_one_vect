from torch import load
import sys

model_name = sys.argv[1]

device = "cpu"
model = load(model_name, map_location=device)

print(model)