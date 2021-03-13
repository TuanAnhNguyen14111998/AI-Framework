from zipfile import ZipFile
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to file compress for dataset (.zip)")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to fordel store dataset")
args = vars(ap.parse_args())

input_compress = args["input"]
output_fordel = args["output"]

# opening the zip file in READ mode 
with ZipFile(input_compress, 'r') as zip: 
    # printing all the contents of the zip file 
    zip.printdir() 
  
    # extracting all the files 
    print('Extracting all the files now...') 
    zip.extractall(output_fordel) 
    print('Done!') 
