## Build the Docker image

Download the [Dockerfile][Dockerfile] to your working directory, navigate to that directory, and build the Docker image:

```shell
docker build --no-cache -f Dockerfile -t mammography-nils .
```
This creates a Docker image named mammography-nils

## Prepare input data and model files
1. Prepare the input mammography images in DICOM format using the same directory structure as the provided test-image folder.
2. Download the Double-CV model files to your local filesystem.
3. Place the input images and model files under a local directory that can be mounted into the Docker container.

For example:
```
{LOCAL_PATH_TO_IMAGE_AND_MODEL}/
├── models/
└── images/
```

## Run the docker image in a container and mount the local directory to the container


```shell
docker run \
    --gpus all \ 
    --shm-size=8g \
    -it \
    --mount type=bind,source={LOCAL_PATH_TO_IMAGE&MODEL},target=/in_out \
    mammography-nils \
    bash
```
The mounted /in_out directory can be used for both input files and generated output files. Files written to /in_out inside the container will also be available in the corresponding local directory on the host machine.

## Evaluate models on a DICOM dataset
Once inside the Docker container, copy the model files and input images from the mounted /in_out directory to the locations expected by the evaluation pipeline.

For example:
```shell
cp -r /in_out/{MODEL_FOLDER} /workspace/mammography-nils
cp -r /in_out/{IMAGE_FOLDER} /workspace/mammography-nils/projects/Mammography/DM_preprocess/

```
Alternatively, you can keep the files under /in_out and update the relevant configuration variables in the test_pipeline.sh, such as INPUT_CSV and DATASET_ROOT, so that they point to the corresponding paths under /in_out.

The run the evaluation pipeline:
```shell
bash projects/Mammography/scripts/test_pipeline.sh
```

