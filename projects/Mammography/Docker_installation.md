## Build the Docker image

Download the [Dockerfile](Dockerfile) to your working directory, navigate to that directory, and build the Docker image:

```shell
docker build --no-cache -f Dockerfile -t mammography-nils .
```
This creates a Docker image named `mammography-nils`.

## Prepare input data and model files
1. Prepare the input mammography images in DICOM format using the same directory structure as the provided `test_examples` folder.
For example:
```
test_examples/
├── dicom/*.dcm
└── meta.csv
```
For meta.csv with DICOM data, the pipeline expects `dicom_path` and `patient_id`. For png dataset, the pipeline expects `png_path` and `patient_id`.
`manufacturer` is optional to refine the preprocessing (to surpress artifacts and to crop background) using vandor-specific hyperparameters.
If `manufacturer` is unknown, the pipeline will use default settings, which has mainly been tested on GE and Seimens mammograms.
The pipeline supports both producing patient-level predictions for LN probility (from 0 to 1) and tumor size (normalized), and producing cohort-level
performance metrics. To enable cohort-level calculation, ground-truth values (`LNM` and `Tsize`) need to be provided in the `meta.csv`.
Valid values for `LNM` include (P,N) or (1,0). Valid values for `Tsize` include numerical variables.
Ground-truth values will never not be used by the models to make predictions. If not available, `LNM` and `Tsize` will be filled with null.
`test_examples/meta.csv` is without ground-truth columns, and  `meta_with_GT.csv` is with ground-truth columns.

2. Download the Double-CV model files to your local filesystem.
3. Place the input images and model files under a local directory that can be mounted into the Docker container.

For example:
```
{LOCAL_PATH_TO_IMAGE_AND_MODEL}/
├── MODEL_FOLDER/
└── IMAGE_FOLDER/
```

## Run the docker image in a container and mount the local directory to the container


```shell
docker run \
    --gpus all \ 
    --shm-size=8g \
    -it \
    --mount type=bind,source={LOCAL_PATH_TO_IMAGE_AND_MODEL},target=/in_out \
    mammography-nils \
    bash
```
The mounted `/in_out` directory can be used for both input files and generated output files. Files written to `/in_out` inside the container will also be available in the corresponding local directory on the host machine.

## Evaluate models on a DICOM dataset
Once inside the Docker container, the model files and input images are mounted to `/in_out` directory.
The run the evaluation pipeline:
```shell
bash projects/Mammography/scripts/test_pipeline.sh /in_out/{IMAGE_FOLDER}/meta.csv /in_out/{OUTPUT_FOLDER} /in_out/{MODEL_FOLDER}
````
OUTPUT_FOLDER is decided by the user.

