import argparse
import os
import pandas as pd
import pickle
import scipy
import numpy as np
from sklearn.utils import resample
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, roc_curve, precision_recall_curve
from scipy.stats import pearsonr
import json



MULTI_TASKS=['multifocality', 'LVI', 'tumor_size', 'N', 'NumPos']

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble across k-fold models prediction and across multiple mammos'
                                                 'for each patients')
    parser.add_argument('work_dir', help='the directory to the predictions')
    parser.add_argument('--out-file',type=str,default='test_patient_AUCs.json')
    parser.add_argument('--double-CV', action='store_true', help='enable ensemble over double cross-validation')
    parser.add_argument('--k-fold', type=int, default=5, help='CV fold numbers')
    parser.add_argument('--bootstrap-repeat-num', type=int, default=1000,help='repeat numbers of bootstrap sampling')
    parser.add_argument('--multi-prediction', action='store_true', help='enable parsing multi-task predictions')
    parser.add_argument('--single-endpoint', type=str, default='N', help='the single task to investigate on')
    args = parser.parse_args()
    return args


def load_prediction_1cv(args, task):
    k_fold=args.k_fold
    print(f"loading {k_fold}-fold cross-validation test predictions {task}")

    data=[]
    for fd1 in range(k_fold):
        print(f"loading file, {args.work_dir}/{fd1}/{fd2}/predict.pkl")
        predictions = pickle.load(open(f"{args.work_dir}/{fd1}/{fd2}/predict.pkl", "rb"))
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
                fortnr = imagename.split("_")[0]
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
            if os.path.exists(f"{args.work_dir}/{fd1}/{fd2}/predict.pkl"):
                print(f"loading file, {args.work_dir}/{fd1}/{fd2}/predict.pkl")
                predictions=pickle.load(open(f"{args.work_dir}/{fd1}/{fd2}/predict.pkl","rb"))
            else:
                raise ValueError(f"{args.work_dir}/{fd1}/{fd2}/predict.pkl does not exist!!!")
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
                        fortnr=imagename.split("_")[0]
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
               'r2':r2_score,
               'pearsonr':pearsonr}
    ms=[]
    bootstrap_sample_size=len(df)
    for i in range(repeat_num):
        fortnrs, target, predict = resample(df.fortnr.values, df.gt_label.values, df.predict.values, replace=True,
                                            n_samples=bootstrap_sample_size, stratify=df.gt_label.values)
        if metric!='pearsonr':
            ms.append(evaluator[metric](target,predict))
        else:
            ms.append(evaluator[metric](target, predict)[0])
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
        return {"r2":-9,"pearsonr":-9}, {"r2":{"mean":-9, "std":-9},"pearsonr":{"mean":-9, "std":-9}}
    r2=r2_score(data[task].gt_label, data[task].predict)
    r2_mean, r2_std=bootstrap_metrics(data[task],bootstrap_repeat_num,'r2')
    prr=pearsonr(data[task].gt_label, data[task].predict)[0]
    prr_mean, prr_std = bootstrap_metrics(data[task], bootstrap_repeat_num, 'pearsonr')
    return {"r2":r2,"pearsonr":prr}, {"r2":{"mean":r2_mean, "std":r2_std},"pearsonr":{"mean":prr_mean, "std":prr_std}}


def auc_curve_result(data,task):
    if data[task].empty:
        return {"roc": {}, "pr": {}}
    r1, r2, thred = roc_curve(data[task].gt_label,data[task].predict)
    assert len(r1)==len(r2)
    roc_points={'x':r1.tolist(),'y':r2.tolist(),'thred':thred.tolist()}

    r1, r2, thred = precision_recall_curve(data[task].gt_label,data[task].predict)
    assert len(r1)==len(r2)
    pr_points={'x':r2.tolist(),'y':r1.tolist(),'thred':thred.tolist()}
    return {'roc':roc_points,'pr':pr_points}

def main():
    args = parse_args()

    if not os.path.exists(args.work_dir):
        raise ValueError(f"Invalid input!!! {args.work_dir} doesn't exists!")

    # load prediction and ensemble over folds and multi-views
    data = load_multi_prediction(args) if args.multi_prediction else load_single_prediction(args)


    data_dict={'patient-level prediction':data}

    # overall performance metrics, AUCs for BC, r2 for regression tasks
    overall={}
    #bootstrap results
    bootstrap={}
    #AUC_curve points
    auc_curves={}
    bootstrap_repeat_num=args.bootstrap_repeat_num
    for datasetname, data in data_dict.items():
        overall[datasetname]={}
        bootstrap[datasetname]={}
        auc_curves[datasetname]={}
        for task in data.keys():
            print(f"{datasetname} has {len(data[task])} patients for {task} prediction")
            if task in ['multifocality', 'LVI', 'N']:
                task_result_overall, task_result_bootstrap=binary_classify_result(data,task,bootstrap_repeat_num)
                overall[datasetname][task]=task_result_overall
                bootstrap[datasetname][task]=task_result_bootstrap
                print(
                    f"{datasetname}, {task} prediction| roc :{task_result_overall['roc']}, "
                    f"pr :{task_result_overall['pr']} \n"
                    f"Bootstrap result|", task_result_bootstrap)
                auc_curves[datasetname][task]=auc_curve_result(data,task)

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
                        f"{datasetname}, {task} r2 :{task_result_overall['r2']} , pearsonr: {task_result_overall['pearsonr']}")
            else:
                task_result_overall, task_result_bootstrap = regress_result(data, task, bootstrap_repeat_num)
                overall[datasetname][task] = task_result_overall
                bootstrap[datasetname][task] = task_result_bootstrap
                print(
                    f"{datasetname}, {task} r2 :{task_result_overall['r2']}, pearsonr: {task_result_overall['pearsonr']} ")

    results={"overall":overall,"bootstrap":bootstrap,
             "patient level prediction":{'predicts':data_dict['patient-level prediction']['N'].predict.values.tolist(),
                                     'patient_ids':data_dict['patient-level prediction']['N'].fortnr.values.tolist(),
                                     'gt_labels':data_dict['patient-level prediction']['N'].gt_label.values.tolist()},}
    with open(f"{args.work_dir}/{args.out_file}","w") as f:
        f.write(json.dumps(results))


if __name__ == '__main__':
    main()
