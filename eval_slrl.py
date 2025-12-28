#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, re, csv, sys
from pathlib import Path
from typing import List, Dict, Set, Tuple

# --- Helper Functions ---
def read_lines(path: Path) -> List[str]:
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        return [ln.strip() for ln in f if ln.strip() != '']

def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in re.findall(r'-?\d+', text)]

def load_pred(res_dir: Path, dataset: str) -> Tuple[List[int], List[int], List[List[int]]]:
    seed_p  = res_dir / f"{dataset}_seed.txt"
    cidx_p  = res_dir / f"{dataset}_com_index.txt"
    pred_p  = res_dir / f"{dataset}_pred_com.txt"
    
    if not (seed_p.exists() and cidx_p.exists() and pred_p.exists()):
        raise FileNotFoundError(f"Missing files in {res_dir}. Need: _seed.txt, _com_index.txt, _pred_com.txt")

    # Load Seeds
    seeds = []
    for ln in read_lines(seed_p):
        seeds += [int(x) for x in re.findall(r'-?\d+', ln)]

    # Load Predicted Community IDs (Ground Truth IDs corresponding to seeds)
    cidxs = []
    for ln in read_lines(cidx_p):
        cidxs += [int(x) for x in re.findall(r'-?\d+', ln)]

    # Load Predicted Nodes
    pred_lines = read_lines(pred_p)
    preds = [[int(x) for x in re.findall(r'-?\d+', line)] for line in pred_lines]

    # Align lengths
    L = min(len(seeds), len(cidxs), len(preds))
    if L < len(seeds):
        print(f"[WARN] Mismatch lengths: seeds={len(seeds)}, com_idxs={len(cidxs)}, preds={len(preds)}. Trimming to {L}.")
    
    return seeds[:L], cidxs[:L], preds[:L]

def load_gt_cmty(cmty_file: Path) -> Dict[int, Set[int]]:
    """
    Load ground truth from .cmty file (line-based).
    Key = Line Index (0-based or 1-based depending on your logic).
    IMPORTANT: Current logic uses 0-based index (Line 1 is ID 0).
    """
    gt = {}
    with cmty_file.open('r', encoding='utf-8', errors='ignore') as f:
        for i, ln in enumerate(f):
            ln = ln.strip()
            if not ln: continue
            nodes = set(parse_int_list(ln))
            gt[i] = nodes # <--- Line 0 is ID 0. Check if your dataset uses 1-based IDs!
    return gt

def load_gt_labels(labels_csv: Path) -> Dict[int, Set[int]]:
    gt = {}
    with labels_csv.open('r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader, None) # Skip header if exists
        for row in reader:
            if not row: continue
            try:
                # Try standard: node_id, com_id
                node_val = int(row[0])
                com_val  = int(row[1])
                gt.setdefault(com_val, set()).add(node_val)
            except:
                pass
    return gt

def prf_jaccard(y_true: Set[int], y_pred: Set[int]) -> Tuple[float,float,float,float,int,int,int]:
    tp = len(y_true & y_pred)
    fp = len(y_pred - y_true)
    fn = len(y_true - y_pred)
    
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec / (prec + rec) if (prec + rec) else 0.0
    jac  = tp / len(y_true | y_pred) if (y_true or y_pred) else 0.0
    
    return prec, rec, f1, jac, tp, fp, fn

# --- Main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--res_dir', default='./res')
    ap.add_argument('--gt_cmty', default=None)
    ap.add_argument('--gt_labels', default=None)
    ap.add_argument('--out_csv', default=None)
    ap.add_argument('--verbose', action='store_true', help="Print details for every seed")
    args = ap.parse_args()

    res_dir = Path(args.res_dir)
    
    print(f"--- Loading Predictions from {res_dir} ---")
    seeds, com_idxs, preds = load_pred(res_dir, args.dataset)

    print(f"--- Loading Ground Truth ---")
    if args.gt_cmty:
        print(f"Loading GT from Community File: {args.gt_cmty}")
        gt = load_gt_cmty(Path(args.gt_cmty))
    elif args.gt_labels:
        print(f"Loading GT from Labels CSV: {args.gt_labels}")
        gt = load_gt_labels(Path(args.gt_labels))
    else:
        raise SystemExit("Error: Please provide --gt_cmty or --gt_labels")
    
    # Check if GT is empty
    if not gt:
        print("[ERROR] Ground Truth dictionary is EMPTY. Check your GT file path or format.")
        return

    rows = []
    print(f"\nEvaluating {len(seeds)} seeds...")
    
    # Statistic tracking
    empty_true_count = 0
    
    for i, (seed, cid, pred_nodes) in enumerate(zip(seeds, com_idxs, preds)):
        y_true = gt.get(cid, set())
        y_pred = set(pred_nodes)
        
        # --- DEBUGGING LOGIC START ---
        # Nếu không tìm thấy Ground Truth cho ID này
        if len(y_true) == 0:
            empty_true_count += 1
            if empty_true_count <= 5: # Chỉ in 5 lỗi đầu tiên để tránh spam
                print(f"\n[DEBUG] Seed={seed}: Cannot find GT for com_id={cid}!")
                print(f"        -> Prediction size: {len(y_pred)} nodes.")
                print(f"        -> Hint: Is com_id 0-based or 1-based? Sample GT keys: {list(gt.keys())[:5]}")

        p,r,f1,j, tp,fp,fn = prf_jaccard(y_true, y_pred)
        
        # Nếu F1 thấp nhưng True Size > 0, in ra so sánh
        if f1 < 0.1 and len(y_true) > 0 and args.verbose:
             print(f"\n[LOW SCORE] Seed={seed} | F1={f1:.4f}")
             print(f"   Pred (First 10): {list(y_pred)[:10]}...")
             print(f"   True (First 10): {list(y_true)[:10]}...")
        # --- DEBUGGING LOGIC END ---

        if args.verbose:
             print(f"[{i:03d}] seed={seed} cid={cid} | true_sz={len(y_true)} pred_sz={len(y_pred)} | F1={f1:.4f}")

        rows.append([i, seed, cid, len(y_true), len(y_pred), p, r, f1, j, tp, fp, fn])

    if empty_true_count > 0:
        print(f"\n[WARNING] Found {empty_true_count} seeds where Ground Truth size = 0.")
        print("          This means the 'com_index' in your results does not match IDs in your GT file.")

    out_csv = Path(args.out_csv) if args.out_csv else (res_dir / f"{args.dataset}_metrics.csv")
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['idx','seed','gt_cid','true_size','pred_size','precision','recall','f1','jaccard','tp','fp','fn'])
        w.writerows(rows)

    # Averages
    if rows:
        import numpy as np
        arr = np.array([[r[5],r[6],r[7],r[8]] for r in rows], dtype=float)
        mean = arr.mean(axis=0)
        print(f"\n=== FINAL AVERAGES ===")
        print(f"Precision : {mean[0]:.4f}")
        print(f"Recall    : {mean[1]:.4f}")
        print(f"F1-Score  : {mean[2]:.4f}")
        print(f"Jaccard   : {mean[3]:.4f}")
        print(f"Saved metrics to: {out_csv}")

if __name__ == "__main__":
    main()