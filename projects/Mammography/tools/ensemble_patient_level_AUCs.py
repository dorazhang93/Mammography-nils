import argparse
import os
import pandas as pd
import pickle
import scipy
import numpy as np
from sklearn.utils import resample
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
import json



MULTI_TASKS=['multifocality', 'LVI', 'tumor_size', 'N', 'NumPos']

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble across k-fold models prediction and across multiple mammos'
                                                 'for each patients')
    parser.add_argument('work_dir', help='the directory to the predictions')
    parser.add_argument('--input-folder', type=str, default='test', help='prediction folder')
    parser.add_argument('--out-file',type=str,default='test_patient_AUCs.json')
    parser.add_argument('--double-CV', action='store_true', help='enable ensemble over double cross-validation')
    parser.add_argument('--k-fold', type=int, default=5, help='CV fold numbers')
    parser.add_argument('--bootstrap-repeat-num', type=int, default=1000,help='repeat numbers of bootstrap sampling')
    parser.add_argument('--bootstrap-data-uncertainty',action='store_true',help='enable data uncertainty estimation through bootstrap')
    parser.add_argument('--patient-list', type=str, default='None',help='patients list to filter patients out if them are not in the list')
    parser.add_argument('--split-cohort', action='store_true',help='enable evaluation on separate cohorts')
    parser.add_argument('--multi-prediction', action='store_true', help='enable parsing multi-task predictions')
    parser.add_argument('--single-endpoint', type=str, default='N', help='the single task to investigate on')
    parser.add_argument('--Npos-softmax', action='store_true', help='enable softmax for 3 classes classification')
    args = parser.parse_args()
    return args


def load_prediction_1cv(args, task):
    k_fold=args.k_fold
    print(f"loading {k_fold}-fold cross-validation test predictions {task}")

    data=[]
    for fd1 in range(k_fold):
        print(f"loading file, {args.work_dir}/{fd1}/{args.input_folder}/best.pkl")
        predictions = pickle.load(open(f"{args.work_dir}/{fd1}/{args.input_folder}/best.pkl", "rb"))
        for i in range(len(predictions)):
            predict = predictions[i][task] if task in predictions[i].keys() else predictions[i]
            if 'gt_label' in predict:
                label = predict['gt_label'].numpy()[0]
                if task in ['multifocality', 'LVI', 'N']:
                    pre_score = scipy.special.expit(predict['pred_score'].numpy()[0])
                elif task == 'NumPos':
                    if args.Npos_softmax:
                        pre_score = F.softmax(predict['pred_score']).numpy()
                        pre_score = pre_score[1] + pre_score[2]
                        if label == 2:
                            label = 1
                    else:
                        pre_score = scipy.special.expit(predict['pred_score'].numpy()[0])
                else:
                    pre_score = predict['pred_score'].numpy()[0]
                imagename = predict['img_path'].split("/")[-1]
                fortnr = "_".join(imagename.split("_")[:3]) if 'mbtst' in imagename else "_".join(imagename.split("_")[:2])
                data.append([fortnr, imagename, label, pre_score, fd1])
    data=pd.DataFrame(data,columns=['fortnr','imagename','gt_label','predict','5fold'])
    data = data.groupby(['fortnr', 'imagename']).agg(gt_label=('gt_label', 'mean'),
                                                          predict=('predict', 'mean')).reset_index()
    print(f"loaded {task} predictions for {len(data)} images ")
    if task in ['multifocality', 'LVI','N']:
        assert len(data[data.gt_label==0])+len(data[data.gt_label==1])==len(data)

    #TODO
    #filter mammograms rather than the latest screening
    data = data.groupby(['fortnr']).agg(gt_label=('gt_label', 'mean'),
                                                 predict=('predict', 'max')).reset_index()
    print(f"loaded {task} predictions for {len(data)} patients ")


    return data

def load_prediction_2cv(args, task):
    k_fold=args.k_fold
    print(f"loading double {k_fold}-fold cross-validation test predictions {task}")
    data=[]
    for fd1 in range(k_fold):
        for fd2 in range(k_fold):
            if os.path.exists(f"{args.work_dir}/{fd1}/{args.input_folder}/best_rd{fd2}.pkl"):
                print(f"loading file, {args.work_dir}/{fd1}/{args.input_folder}/best_rd{fd2}.pkl")
                predictions=pickle.load(open(f"{args.work_dir}/{fd1}/{args.input_folder}/best_rd{fd2}.pkl","rb"))
            else:
                print(f"loading file, {args.work_dir}/{fd1}/{fd2}/{args.input_folder}.pkl")
                predictions=pickle.load(open(f"{args.work_dir}/{fd1}/{fd2}/{args.input_folder}.pkl","rb"))
            for i in range(len(predictions)):
                predict= predictions[i][task] if task in predictions[i].keys() else predictions[i]
                if 'gt_label' in predict:
                    label=predict['gt_label'].numpy()[0]
                    if task in ['multifocality', 'LVI','N']:
                        pre_score=scipy.special.expit(predict['pred_score'].numpy()[0])
                    elif task == 'NumPos':
                        if args.Npos_softmax:
                            pre_score=F.softmax(predict['pred_score']).numpy()
                            pre_score=pre_score[1]+pre_score[2]
                            if label==2:
                                label=1
                        else:
                            pre_score = scipy.special.expit(predict['pred_score'].numpy()[0])
                    else:
                        pre_score=predict['pred_score'].numpy()[0]
                    if 'img_path' in predict:
                        imagename=predict['img_path'].split("/")[-1]
                        fortnr="_".join(imagename.split("_")[:3]) if 'mbtst' in imagename else "_".join(imagename.split("_")[:2])
                    else:
                        imagename='Unknown'
                        fortnr=predict['fortnr']
                    data.append([fortnr, imagename,label,pre_score,fd1,fd2])
    data=pd.DataFrame(data,columns=['fortnr','imagename','gt_label','predict','5fold','5run'])
    data = data.groupby(['fortnr', 'imagename']).agg(gt_label=('gt_label', 'mean'),
                                                          predict=('predict', 'mean')).reset_index()
    print(f"loaded {task} predictions for {len(data)} images ")
    if task in ['multifocality', 'LVI','N']:
        assert len(data[data.gt_label==0])+len(data[data.gt_label==1])==len(data)
    #TODO
    #filter mammograms rather than the latest screening
    data = data.groupby(['fortnr']).agg(gt_label=('gt_label', 'mean'),
                                                 predict=('predict', 'max')).reset_index()
    print(f"loaded {task} predictions for {len(data)} patients ")
    return data

def load_multi_prediction(args):
    multi_data={}
    print(f"loading dumped multi predictions from {args.work_dir}")
    for task in MULTI_TASKS:
        data=load_prediction_2cv(args,task) if args.double_CV else load_prediction_1cv(args,task)
        multi_data[task]=data

    return multi_data

def load_single_prediction(args):
    single_data={}
    task=args.single_endpoint
    print(f"loading dumped {task} predictions from {args.work_dir}")

    assert task in MULTI_TASKS
    single_data[task]=load_prediction_2cv(args,task) if args.double_CV else load_prediction_1cv(args,task)
    return single_data

def bootstrap_metrics(df, repeat_num, metric):
    evaluator={"roc":roc_auc_score,
               'pr':average_precision_score,
               'r2':r2_score}
    ms=[]
    bootstrap_sample_size=len(df)
    for i in range(repeat_num):
        fortnrs, target, predict = resample(df.fortnr.values, df.gt_label.values, df.predict.values, replace=True,
                                            n_samples=bootstrap_sample_size, stratify=df.gt_label.values)
        ms.append(evaluator[metric](target,predict))
    return np.array(ms).mean(), np.array(ms).std()

def binary_classify_result(data, task, bootstrap_repeat_num):
    if data[task].empty:
        return {"roc": -1, "pr": -1}, {"roc": {"mean": -1, "std": -1},
                                    "pr": {"mean": -1, "std": -1}}

    roc = roc_auc_score(data[task].gt_label, data[task].predict)
    pr = average_precision_score(data[task].gt_label, data[task].predict)

    roc_mean, roc_std = bootstrap_metrics(data[task], bootstrap_repeat_num, 'roc')
    pr_mean, pr_std = bootstrap_metrics(data[task], bootstrap_repeat_num, 'pr')

    return {"roc": roc, "pr": pr}, {"roc": {"mean": roc_mean, "std": roc_std},
                                    "pr": {"mean": pr_mean, "std": pr_std}}

def regress_result(data, task, bootstrap_repeat_num):
    if data[task].empty:
        return {"r2":-9}, {"r2":{"mean":-9, "std":-9}}
    r2=r2_score(data[task].gt_label, data[task].predict)
    r2_mean, r2_std=bootstrap_metrics(data[task],bootstrap_repeat_num,'r2')
    return {"r2":r2}, {"r2":{"mean":r2_mean, "std":r2_std}}


def average_aucs(args):
    k_fold=args.k_fold

    rocs=[]
    prs=[]
    for fd2 in range(k_fold):
        data=[]
        for fd1 in range(k_fold):
            print(f"loading file, {args.work_dir}/{fd1}/test/best_rd{fd2}.pkl")
            predictions = pickle.load(open(f"{args.work_dir}/{fd1}/test/best_rd{fd2}.pkl", "rb"))
            for i in range(len(predictions)):
                predict = predictions[i]
                if 'gt_label' in predict:
                    label = predict['gt_label'].numpy()[0]
                    pre_score = scipy.special.expit(predict['pred_score'].numpy()[0])
                    imagename = predict['img_path'].split("/")[-1]
                    fortnr = imagename.split('_')[1] if 'cohort2' in imagename else int(imagename.split('_')[0])
                    data.append([fortnr, imagename, label, pre_score, fd1])
        data = pd.DataFrame(data, columns=['fortnr', 'imagename', 'gt_label', 'predict', '5fold'])
        data = data.groupby(['fortnr', 'imagename']).agg(gt_label=('gt_label', 'mean'),
                                                         predict=('predict', 'mean')).reset_index()
        assert len(data[data.gt_label == 0]) + len(data[data.gt_label == 1]) == len(data)

        data = data.groupby(['fortnr']).agg(gt_label=('gt_label', 'mean'),
                                            predict=('predict', 'max')).reset_index()
        print(f"loaded predictions for {len(data)} patients ")
        rocs.append(roc_auc_score(data.gt_label, data.predict))
        prs.append(average_precision_score(data.gt_label, data.predict))
    print(f"average over 5 folds: roc {np.array(rocs).mean()}, {np.array(rocs).std()}, "
          f"pr {np.array(prs).mean()}, {np.array(prs).std()}")
    return

def main():
    args = parse_args()

    if not os.path.exists(args.work_dir):
        raise ValueError(f"Invalid input!!! {args.work_dir} doesn't exists!")

    # load prediction and ensemble over folds and multi-views
    data = load_multi_prediction(args) if args.multi_prediction else load_single_prediction(args)

    # average_aucs(args)

    data_dict={}
    if args.patient_list !="None":
        print(f"select patients using {args.patient_list}")
        fortnrs_list=np.loadtxt(args.patient_list, dtype='str')
        mbtst={}
        for k,v in data.items():
            data[k]=v[v.fortnr.isin(fortnrs_list.tolist())].reset_index(drop=True)
            mbtst[k]=v[v.fortnr.str.startswith("mbtst")].reset_index(drop=True)
            print(f" mbtst has {len(mbtst[k])} patients")
        data_dict['mbtst']=mbtst
    data_dict["all"]=data
    if args.split_cohort:
        print(f"split predictions into cohort1 and cohort2")
        data_c1={}
        data_c2={}
        for task, df in data.items():
            c1_idx=np.array([True if 'cohort1' in fortnr else False for fortnr in df.fortnr.values])
            data_c1[task]=df[c1_idx].reset_index(drop=True)
            data_c2[task]=df[~c1_idx].reset_index(drop=True)
            print(f"{task}: cohort1 has {len(data_c1[task])} patients, cohort2 has {len(data_c2[task])} patients")
        data_dict['cohort1']=data_c1
        data_dict['cohort2']=data_c2


    # overall performance metrics, AUCs for BC, r2 for regression tasks
    overall={}
    #bootstrap results
    bootstrap={}
    bootstrap_repeat_num=args.bootstrap_repeat_num
    for datasetname, data in data_dict.items():
        overall[datasetname]={}
        bootstrap[datasetname]={}
        for task in data.keys():
            if task in ['multifocality', 'LVI', 'N']:
                task_result_overall, task_result_bootstrap=binary_classify_result(data,task,bootstrap_repeat_num)
                overall[datasetname][task]=task_result_overall
                bootstrap[datasetname][task]=task_result_bootstrap
                print(
                    f"{datasetname}, {task} roc :{task_result_overall['roc']}, "
                    f"pr :{task_result_overall['pr']}")

            elif task == 'NumPos':
                if args.Npos_softmax:
                    task_result_overall, task_result_bootstrap = binary_classify_result(data, task,
                                                                                        bootstrap_repeat_num)
                    overall[datasetname][task] = task_result_overall
                    bootstrap[datasetname][task] = task_result_bootstrap
                    print(
                        f"{datasetname}, {task} roc :{task_result_overall['roc']}, "
                        f"pr :{task_result_overall['pr']}")

                else:
                    task_result_overall, task_result_bootstrap = regress_result(data, task, bootstrap_repeat_num)
                    overall[datasetname][task] = task_result_overall
                    bootstrap[datasetname][task] = task_result_bootstrap
                    print(
                        f"{datasetname}, {task} r2 :{task_result_overall['r2']} ")
            else:
                task_result_overall, task_result_bootstrap = regress_result(data, task, bootstrap_repeat_num)
                overall[datasetname][task] = task_result_overall
                bootstrap[datasetname][task] = task_result_bootstrap
                print(
                    f"{datasetname}, {task} r2 :{task_result_overall['r2']} ")

    results={"overall":overall,"bootstrap":bootstrap}
    with open(f"{args.work_dir}/{args.out_file}","w") as f:
        f.write(json.dumps(results))


if __name__ == '__main__':
    main()
