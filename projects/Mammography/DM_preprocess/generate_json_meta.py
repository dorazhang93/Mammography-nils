import os, json
import argparse
import pandas as pd
from pathlib import Path
from dm_preprocess import DMImagePreprocessor
processor = DMImagePreprocessor()
from collections import defaultdict
import pydicom as dicom
import cv2

def parse_args():
    parser=argparse.ArgumentParser(description="Generate json meta file for test cohort")
    parser.add_argument('--input-csv', type=str, help="File path to the input csv txt file")
    parser.add_argument('--input-image-format', help="dicom or png")
    parser.add_argument('--output-folder',type=str, help="Folder path to the output json meta file and png images")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()

    if os.path.exists(args.input_csv):
        print(f"Loading meta csv file from ", args.input_csv)
        meta_df = pd.read_csv(args.input_csv) #  columns including 'dicom_path', 'patient_id', 'view', 'laterality', 'LNM'
    else:
        raise ValueError(f"{args.input_csv} does not exist!!!")
    meta_df=meta_df.replace({'LNM':{'P':1,'N':0}})

    Path(args.output_folder).mkdir(exist_ok=True, parents=True)
    png_folder = f"{args.output_folder}/png_cropped_breast"
    Path(png_folder).mkdir(exist_ok=True, parents=True)
    print(f"Png images will be save to {png_folder}")

    meta_dict = {'data_list':[]}
    png_store=defaultdict(list)
    for i in range(len(meta_df)):
        patient_id = meta_df.loc[i,'patient_id']
        view = meta_df.loc[i,'view']
        laterality = meta_df.loc[i,'laterality']
        ln = meta_df.loc[i, 'LNM']
        manufacturer = meta_df.loc[i, 'manufacturer']
        if args.input_image_format == 'png':
            raw_image_path=meta_df.loc[i, 'png_path']
            image = cv2.imread(raw_image_path)
        elif args.input_image_format == 'dicom':
            raw_image_path=meta_df.loc[i, 'dicom_path']
            image = dicom.dcmread(raw_image_path)
        else:
            raise ValueError(f"Invalid input image format {args.input_image_format}!!!")

        if manufacturer=='HOLOGIC':#TODO check the vendor name and valid below segment parameters on more real images
            image_processed, breast_bbox = processor.process(image,
                                                         artif_suppression=True,
                                                         segment_breast=True,
                                                             segment_low_int_threshold=.01,
                                                             segment_erode_kernel=5,
                                                             segment_n_erode=5)

        else:
            image_processed, breast_bbox = processor.process(image,
                                                         artif_suppression=True,
                                                         segment_breast=True)


        if image_processed is None:
            print("Invalid dicom data!!!", raw_image_path)
            continue
        else:
            image_processed = image_processed[breast_bbox[1]:breast_bbox[1] + breast_bbox[3], breast_bbox[0]:breast_bbox[0] + breast_bbox[2]]

        png_imagename = f"{png_folder}/{patient_id}_{view}_{laterality}"

        # add visit number to image name
        png_store[png_imagename].append(1)
        png_imagename = f"{png_imagename}_visit{sum(png_store[png_imagename])}.png"

        cv2.imwrite(png_imagename, image_processed)
        meta_dict['data_list'].append({'img_path': png_imagename,
                                       'gt_label':{'N':int(ln)},
                                       'clinic_vars':None}
                                      )

    with open(f"{args.output_folder}/test.json","w") as f:
        json.dump(meta_dict,f)
        print(f"DONE! meta file of {len(meta_dict['data_list'])} mammograms was saved to {args.output_folder}/test.json")

