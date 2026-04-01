## Prepare working path and data/model files for building docker image
```shell
mkdir Mammography-nils
cd Mammography-nils
mkdir docker
mv Dockerfile docker/
mv {TEST_IMAGE_DIR} projects/Mammography/DM_preprocess/
mv {MODEL_DIR} .
```
## Installation using Dockerfile

Below are quick steps for installation:

```shell
docker build --no-cache -f docker/Dockerfile -t mammography-nils .
docker run --gpus all --shm-size=8g -it mammography-nils bash
```


## Evaluate models on dicom dataset

```shell
bash projects/Mammography/scripts/test_pipeline.sh
```
