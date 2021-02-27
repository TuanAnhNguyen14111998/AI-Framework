import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to file fordel origin dataset")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to fordel save file csv information split dataset")
ap.add_argument("-op", "--option", type=str, default="number",
	help="path to fordel save file csv information split dataset")

ap.add_argument("-n_train", "--number_train", type=int, default=0,
	help="percent of training data")
ap.add_argument("-n_val", "--number_val", type=int, default=0,
	help="percent of validation data")
ap.add_argument("-n_test", "--number_test", type=int, default=0,
	help="percent of testing data")

ap.add_argument("-p_train", "--percent_train", type=float, default=0,
	help="percent of training data")
ap.add_argument("-p_val", "--percent_val", type=float, default=0,
	help="percent of validation data")
ap.add_argument("-p_test", "--percent_test", type=float, default=0,
	help="percent of testing data")
args = vars(ap.parse_args())

input_fordel = args["input"]
output_fordel = args["output"]
option = args["option"]

number_train = args["number_train"]
number_val = args["number_val"]
number_test = args["number_test"]

percent_train = args["percent_train"]
percent_val = args["percent_val"]
percent_test = args["percent_test"]

print("Starting split dataset ... ")
df = pd.read_csv(output_fordel + "/information_dataset.csv")
list_class_name = df.class_name.unique()

record_by_class_name = {}
for class_name in list_class_name:
    record_by_class_name[class_name] = df[df["class_name"] == class_name]

info_train_set = output_fordel + "/information_training.csv"
info_val_set = output_fordel + "/information_validation.csv"
if number_test != 0 or percent_test != 0:
    info_test_set = output_fordel + "/information_testing.csv"

df_train = None
df_val = None
df_test = None
index = 0


for class_name in tqdm(list_class_name):
    df_class = record_by_class_name[class_name]
    if option == "number":
        if number_test == 0:
            if len(df_class) > number_train:
                train = df_class.sample(number_train)
                val = df_class.drop(train.index)
            else:
                train, val = \
                    np.split(df_class.sample(frac=1, random_state=42), 
                            [int(.8*len(df_class))])
            df_train = pd.concat([df_train, train])
            df_val = pd.concat([df_val, val])
        else:
            # xu ly sau
            train, val, test = \
              np.split(df_class.sample(frac=1, random_state=42), 
                       [int(.8*len(df_class)), int(.9*len(df_class))])
            df_train = pd.concat([df_train, train])
            df_val = pd.concat([df_val, val])
            df_test = pd.concat([df_test, test])
    else:
        if percent_test == 0:
            train, val = \
                np.split(df_class.sample(frac=1, random_state=42), 
                        [int(percent_train*len(df_class))])
            df_train = pd.concat([df_train, train])
            df_val = pd.concat([df_val, val])
        else:
            train, val, test = \
                np.split(df_class.sample(frac=1, random_state=42), 
                        [int(percent_train*len(df_class)), 
                         int((percent_train+percent_val)*len(df_class))])
            df_train = pd.concat([df_train, train])
            df_val = pd.concat([df_val, val])
            df_test = pd.concat([df_test, test])

    # if index > 1:
    #     break
    # index += 1
            
    # import pdb; pdb.set_trace()

df_train.to_csv(info_train_set, index = False, header=True)
df_val.to_csv(info_val_set, index = False, header=True)
if number_test != 0 or percent_test != 0:
    df_test.to_csv(info_test_set, index = False, header=True)

print("Done!")