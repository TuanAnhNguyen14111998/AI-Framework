import pandas as pd
import argparse
import json

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to fordel save file information dataset (csv)")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to fordel save file class info (csv)")
args = vars(ap.parse_args())

input_fordel = args["input"]
output_fordel = args["output"]

print("Writing file information class ... ")

df = pd.read_csv(input_fordel + "/information_dataset.csv")
list_class_name = list(df.class_name.unique())

info_class = {}
for index, class_name in enumerate(list_class_name):
    info_class[class_name] = index

with open(output_fordel + "/information_class.json", 'w') as f:
    json.dump(info_class, f)

print("Done!")
