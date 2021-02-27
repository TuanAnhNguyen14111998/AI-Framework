from tqdm import tqdm
import os
import argparse
import glob
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to fordel dataset")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to fordel save file csv")
args = vars(ap.parse_args())

input_fordel = args["input"]
output_fordel = args["output"]

string_match = input_fordel + "/*/*"
list_image = glob.glob(string_match)

index = 1
with open(output_fordel + "/information_dataset.csv", mode='w') as csv_file:
    print("Writing file information_dataset.csv ... ")
    fieldnames = ['index', 'file_name', 'class_name', 'image_path']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for image_path in tqdm(list_image):
        file_name = image_path.split("/")[-1]
        class_name = image_path.split("/")[-2]

        writer.writerow({
            'index': index, 
            'file_name': file_name, 
            'class_name': class_name,
            'image_path': image_path})
        index += 1
    
    csv_file.close()
    print("Done!")
