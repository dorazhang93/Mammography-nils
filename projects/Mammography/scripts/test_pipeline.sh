#!/usr/bin/env bash
# 1. set up png images and meta file
echo "@@@@@ Step 1: run image preprocessing to generate png files with cropped breasts and a json meta file @@@@@"
INPUT_CSV="projects/Mammography/DM_preprocess/test_examples/meta.csv"
DATASET_ROOT="/HOME_FOLDER/mmpretrain/projects/Mammography/DM_preprocess/test_examples" #TODO: need to be absolute path, it is where to save the processed png
python projects/Mammography/DM_preprocess/generate_json_meta.py --input-csv ${INPUT_CSV}  \
  --input-image-format "dicom" \
  --output-folder ${DATASET_ROOT}

# 2. run double-CV models on independent test sets
echo "@@@@@ Step 2: run double-CV models using the generated json meta file @@@@@"
OUTPATH="work_dirs_phase2_unfreeze/fullimage1792x1024_resnet50-expan2_ssl_finetune_transformer_Aug3_barlowtwins_2cv_multitask" # folder to the saved models
EXP="fullimage1792x1024_resnet50-expan2_ssl_finetune_transformer_Aug3"
CUDA_DEVICE=1
PREDICTION_OUT_DIR=${DATASET_ROOT}/prediction
mkdir -p ${PREDICTION_OUT_DIR}
for r1 in {0..4}; do
  out_dir="${OUTPATH}""/""${r1}"
  echo $out_dir
  for r2 in {0..4}; do
    ckpt_dir=${out_dir}/ckpt/rd${r2}
    best_ckpt=$(find ${ckpt_dir} -type f -name 'best*.pth')
    echo "run images on ""$best_ckpt"
    # create folder to save predicitons
    prediction_folder=${PREDICTION_OUT_DIR}/${r1}/${r2}
    mkdir -p ${prediction_folder}

    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python tools/test.py ${out_dir}/${EXP}.py ${best_ckpt} --work-dir work_dirs_test/ \
      --out ${prediction_folder}/predict.pkl \
      --out-item pred \
      --cfg-options test_dataloader.dataset.data_root=${DATASET_ROOT} \
      test_dataloader.dataset.ann_file=test.json \
      log_level=INFO > run_models_log.txt #TODO: to debug, this can be turned to DEBUG
    wait
  done
done
echo "DONE! Prediction probabilities were saved to ""${PREDICTION_OUT_DIR}"

# 3. ensemble patient level prediction and calculate performance AUCs
echo "@@@@@ Step 3: ensemble patient level prediction and calculate performance AUCs @@@@@"
python projects/Mammography/tools/ensemble_patient_level_AUCs_pipeline.py ${PREDICTION_OUT_DIR} --out-file "test_patient_AUCs.json" \
  --double-CV

echo "Performance metrics on patient level using model ensemble were saved to ""${PREDICTION_OUT_DIR}""/test_patient_AUCs.json"
