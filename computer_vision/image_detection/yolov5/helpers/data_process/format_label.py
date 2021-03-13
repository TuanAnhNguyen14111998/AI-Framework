import os
import argparse
import json
from tqdm import tqdm
import glob
from bs4 import BeautifulSoup
import shutil

def create_new_fordel(fordel):
    if os.path.isdir(fordel)==False:
        os.mkdir(fordel)

def format_xml_label(type_set="train"):
    print("\nFormat xml file label to txt file label for {} set .... ".format(type_set))
    if type_set == "train":
        match_str = input_train + "/*.jpg"
    else:
        match_str = input_test + "/*.jpg"
    
    for file_name in tqdm(glob.glob(match_str)):
        file_xml = file_name.replace(".jpg", ".xml")
        with open(file_xml, 'r') as f:
            data = f.read()
        Bs_data = BeautifulSoup(data, "xml")

        width = Bs_data.find("width").text
        height = Bs_data.find("height").text

        if width == "0":
            continue
        
        objects = Bs_data.find_all('object')
        for obj in objects:
            name = obj.find("name").text
            x_min = float(obj.find("xmin").text) / float(width)
            y_min = float(obj.find("ymin").text) / float(height)
            x_max = float(obj.find("xmax").text) / float(width)
            y_max = float(obj.find("ymax").text) / float(height)

            x_center = str((x_min + x_max) / 2)
            y_center = str((y_min + y_max) / 2)
            width_bbx = str((x_max - x_min) / 2)
            height_bbx = str((y_max - y_min) / 2)
            class_id = str(class_dict[name])

            format_label = open(output_save + "/dataset_clean/labels/" + type_set + "/" + os.path.basename(file_xml).replace(".xml", ".txt"), "a")
            string_label = " ".join((class_id, x_center, y_center, width_bbx, height_bbx))
            format_label.write(string_label + "\n")
            format_label.close()
        
        shutil.copyfile(file_name, output_save + "/dataset_clean/images/" + type_set + "/" + os.path.basename(file_name))


if __name__ == "__main__":

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i_train", "--input_train", type=str, default="",
        help="path to fordel save train image origin")
    ap.add_argument("-i_test", "--input_test", type=str, default="",
        help="path to fordel save test image origin")
    ap.add_argument("-op", "--option", type=str, default="xml",
        help="option for format label")
    ap.add_argument("-o", "--fordel_save", type=str, default="xml",
        help="fordel for save image and label format match yolov5")
    args = vars(ap.parse_args())

    option = args["option"]

    input_train = args["input_train"]
    input_test = args["input_test"]

    output_save = args["fordel_save"]

    with open('data/custom_dataset/dataset_info/dataset_info.json') as f:
        class_dict = json.load(f)

    # create new fordel
    create_new_fordel(output_save + "/dataset_clean")

    create_new_fordel(output_save + "/dataset_clean/images")
    create_new_fordel(output_save + "/dataset_clean/images/train")
    create_new_fordel(output_save + "/dataset_clean/images/test")

    create_new_fordel(output_save + "/dataset_clean/labels")
    create_new_fordel(output_save + "/dataset_clean/labels/test")
    create_new_fordel(output_save + "/dataset_clean/labels/train")
    
    # format label and construct fordel for yolov5
    if option == "xml":
        format_xml_label("train")
        format_xml_label("test")
