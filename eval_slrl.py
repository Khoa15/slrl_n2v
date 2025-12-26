#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, re, csv, sys
from pathlib import Path
from typing import Union, List, Dict, Set, Tuple


def prf_jaccard(y_true: Set[int], y_pred: Set[int]) -> Tuple[float,float,float,float,int,int,int]:
    tp = len(y_true & y_pred) # Giao (Intersection)
    fp = len(y_pred - y_true)
    fn = len(y_true - y_pred)
    
    # Tránh chia cho 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    union_len = len(y_true | y_pred) # Hợp (Union)
    jac  = tp / union_len if union_len > 0 else 0.0
    
    return prec, rec, f1, jac, tp, fp, fn

def eval_scores(pred_comm: Union[List, Set],
                    true_comm: Union[List, Set]) -> (float, float, float, float):
    """
    Tính toán Độ chính xác, Thu hồi, F1 và Jaccard
    @param pred_comm: Cộng đồng dự đoán
    @param true_comm: Cộng đồng thực
    """
    intersect = set(true_comm) & set(pred_comm)
    p = len(intersect) / len(pred_comm)
    r = len(intersect) / len(true_comm)
    f = 2 * p * r / (p + r + 1e-9)
    j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
    return round(p, 5), round(r, 5), round(f, 5), round(j, 5)

# --- Main ---
def main():
    datasets = ['amazon', 'dblp'] #, 'lj', 'twitter', 'youtube']

    for dataset in datasets:
        with open(f'./res/{dataset}_com_index.txt', 'r') as f:
            true_cmty_index = int(f.readline().strip())

        with open(f'./res/{dataset}_pred_com.txt', 'r') as f:
            str_pred_cmty = f.readline().strip()

        with open(f'./res/{dataset}_com_index.txt', 'r') as f:
            seed_node = int(f.readline().strip())

        with open(f'./datasets/{dataset}/{dataset}-1.90.cmty.txt', 'r') as f:
            str_true_cmty = f.readlines()[true_cmty_index].strip()

        list_pred = [int(x) for x in str_pred_cmty.split()]
        list_true = [int(x) for x in str_true_cmty.split()]

        prec, rec, f1, jac = eval_scores(list_pred, list_true)

        print(f"Dataset: {dataset}")
        print(f"  Precision: {prec}")
        print(f"  Recall   : {rec}")
        print(f"  F1       : {f1}")
        print(f"  Jaccard  : {jac}")
        print("-" * 20)
        # print(f": {prec}")


        
        


if __name__ == "__main__":
    main()