# Implementation for Mammogram DL pipelines
In this folder, we implemented mammography DL pipelines under the open-mmlab MMPreTrain framework. This work is published and is available at XXXX
## Deep learning on routine full-breast mammograms enhances lymph node metastasis prediction in early breast cancer
In this study, we developed and validated DL models based on preoperatively available clinicopathological predictors and full-breast digital mammogram features to predict LNM status and comprehensively evaluated the added value of routine mammograms to preoperatively predict LNM in multicenter cohorts of cN0 T1–T2 breast cancer patients having primary surgery. The results demonstrate that advances in DL techniques, including domain-adaptive pretraining through SSL and sophisticated spatial modeling empowered by a Transformer, significantly improve imaging analysis, particularly for full-breast mammograms. Our findings reveal that routine mammograms can substantially enhance the preoperative diagnosis of LNM in early-stage breast cancer, and are as informative as key postoperative pathological indicators, such as tumor size and multifocality. Notably, full-breast mammograms exhibited overall comparable predictive performance to tumor region-focused models and showed favorable discriminating ability in the independent test set, underscoring its clinical applicability. 

![Overview of the Pathformer](DL_workflow.png)
## Getting Started

To get a local copy up and running, follow these simple steps

### Prerequisites

python 3.8.18, check environment.yml for list of needed packages

### Installation

1.Clone the repo

```git clone https://github.com/dorazhang93/Mammography-nils.git```

2.Create conda environment

```conda env create --name VIR_ENV_NAME --file=environment.yml --force```

## Usage

### 1.Activate the created conda environment

```conda activate VIR_ENV_NAME```

```cd Mammography-nils```

### 2.Data preprocessing
Prepare unlabeled mammogram patches for self-supervised learning and single- or multi-task labeled mammogram ROIs or full-breast mammograms for supervised learning. Detailed preprocessing steps is available in the publication.
Dataloader API reused the open mmlab design with minor changes. Source code is under the folder -- dataset/

### 3. Self-supervised training of unlabeled mammogram patches
Configuration files for the three state-of-the-art SSL algorithms were placed under the folder -- configs/pretrain

An example to run the BarlowTwins algorithm using 2 GPUs with mixed-precsion mode for efficient training :
bash tools/dist_train.sh configs/pretrain/barlowtwins_resnet50-exp2_-coslr-100e_MMpatches.py 2 --amp > barlowtwins.log

### 4. Supervised training of labeled mammograms

Configuration files for supervised training on multi-task mammograms of full-breast and ROIs were given under the folder configs/multi_task_SL.

An example to run SL on full-breast images using Transformer neck and ResNet backbone pretrained with BarlowTwins SSL:
python projects/Mammography/tools/train.py projects/Mammography/configs/multi_task_SL/fullimage1792x1024_resnet50-expan2_barlowtwin_finetune_transformer_Aug3.py \
      --work-dir ${out_dir} \
      --cfg-options default_hooks.checkpoint.out_dir=${ckpt_dir} \
      train_dataloader.dataset.data_root=${data_root} \
      val_dataloader.dataset.data_root=${data_root} \
      test_dataloader.dataset.data_root=${data_root} \
      train_dataloader.dataset.ann_file=${anno_subfolder}/train.json \
      val_dataloader.dataset.ann_file=${anno_subfolder}/val.json \
      test_dataloader.dataset.ann_file=${anno_subfolder}/meta/test.json

To evaluate above trained model:
python tools/test.py ${out_dir}/${EXP}.py ${best_ckpt} --work-dir work_dirs_test/ --out ${out_dir}/test/best_rd${r2}.json --out-item metrics

To run mammogram features using above trained model:
python tools/test.py ${out_dir}/${EXP}.py ${best_ckpt} --work-dir work_dirs_test/ --out ${out_dir}/test/best_rd${r2}.pkl --out-item pred

### 5. Combining mammogram features and clinical variables

python projects/Mammography/mlp.py --data-root ${data_root} --mode "clinical-mammo" \
          --work-dir ${out_dir} >${out_dir}/out.log