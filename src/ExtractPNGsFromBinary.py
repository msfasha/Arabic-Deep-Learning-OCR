'''
This utility program extracts Arabic words image files from binary files
and stores them as single png files.

The binary repository files can be found at:
# https://drive.google.com/drive/u/2/folders/1mRefmN4Yzy60Uh7z3B6cllyyOXaxQrgg


Each binary file contain a number of word images
For example, the file 1_nice_60000_rows contains 60000 Arabic word images printed in nice font
18_fonts_2160000 samples file contains 2,160,000 Arabic words images printed with 18 different fonts.


To extract images from a binary file, you need to download the binary image file as well as 
the accompanying txt file that contains labels and information about the stored images, set the 
path to these files below and run this program. Make sure to install the required packages i.e. Pillow.


More information about this project can be found at:
https://github.com/msfasha/Arabic-Deep-Learning-OCR

'''
import os
import shutil
import PIL.Image as Image


# Set the path to the binary image files, the txt file contains the labels and the information about
# each image that is stored in the accompanying bin file:
BASE_LABELS_FILE = "/home/me/Downloads/1_nice_60000_rows.txt"
BASE_IMAGES_FILE = "/home/me/Downloads/1_nice_60000_rows.bin"


# Set the path of the output folder i.e. where you want image files to be extracted
OUTPUT_FOLDER = "/home/me/Downloads/pngs/"
FILTERED_LABELS_FILE = OUTPUT_FOLDER + "labels.txt"


# Sample class resemble the structure of information stored about each image stored
# in the related binary file, this includes the start position of the first byte, the size of the
# image, the dimension of the image...etc
class Sample:
    "a single sample from the dataset"

    def __init__(self, gtText, imageIdx, imageHeight, imageWidth, imageSize, imageStartPosition):
        self.gtText = gtText
        self.imageIdx = imageIdx
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.imageSize = imageSize
        self.imageStartPosition = imageStartPosition


samples = []


def main():

    # Delete the output directory if it exists
    print("deleting files in output folder...")
    shutil.rmtree(OUTPUT_FOLDER)

    # recreate output directory
    os.mkdir(OUTPUT_FOLDER)
    print("finished deleting files in output folder")

    # open dataset files
    labelsFile = open(BASE_LABELS_FILE, encoding="utf-8")
    binaryImageFile = open(BASE_IMAGES_FILE, "rb")

    # create a file to store filtered labels i.e. the Arabic word
    filtered_labels_file = open(FILTERED_LABELS_FILE, "w")

    idx = 0

    # extract image information from the labels file and place the result into samples list
    for line in labelsFile:

        lineSplit = line.split(';')

        imgIdx = lineSplit[0]
        imgIdx = imgIdx[10:]

        imgStartPosition = lineSplit[1]
        imgStartPosition = int(imgStartPosition[15:])

        imgHeight = lineSplit[2]
        imgHeight = int(imgHeight[13:])
        imgWidth = lineSplit[3]
        imgWidth = int(imgWidth[12:])
        imgSize = imgHeight * imgWidth

        gtText = lineSplit[8]
        gtText = gtText[5:]

        samples.append(Sample(gtText, imgIdx, imgHeight,
                              imgWidth, imgSize, imgStartPosition))
        filtered_labels_file.write(str(idx) + "_" + gtText)
        idx += 1

    filtered_labels_file.close()
    print(f"Created a labels file with {idx} labels")

    idx = 0

    # extract images from binary file
    print("Extracting images...")

    # Iterate through all the samples in the sample array
    # (sample array contains image information e.g. size, dimenstions...etc)
    for sample in samples:
        # goto the first byte of the image
        binaryImageFile.seek(sample.imageStartPosition)
        # read bytes according to image size
        imgBytes = binaryImageFile.read(sample.imageSize)

        # create image, model is "L", each pixel is represented by a byte
        image = Image.frombytes(
            "L", (sample.imageWidth, sample.imageHeight), imgBytes)
        image.save(OUTPUT_FOLDER + str(idx) + ".png")

        idx += 1

    print(f"Finished extracting {idx} images...")


if __name__ == "__main__":
    main()
