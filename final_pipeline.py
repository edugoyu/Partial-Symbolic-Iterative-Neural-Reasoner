import os
import argparse
import torch
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from utils.common import set_seed, generate_board
from models.transformer_queens import QueensTransformer
from models.transformer_mask_alternative import AlternativeMaskTransformer

def check_symbolic(queen_coords, grid_data, n):
    errors = []
    if len(queen_coords) != n: errors.append(f"Wrong Count ({len(queen_coords)}/{n})")
    rows, cols, colors = [q[1] for q in queen_coords], [q[0] for q in queen_coords], [grid_data[q[1]-1, q[0]-1] for q in queen_coords]
    if len(set(rows)) != len(rows): errors.append("Row Conflict")
    if len(set(cols)) != len(cols): errors.append("Column Conflict")
    if len(set(colors)) != len(colors): errors.append("Color Conflict")
    for i, (x1, y1) in enumerate(queen_coords):
        for j, (x2, y2) in enumerate(queen_coords):
            if i != j and abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                if "Adjacency Violation" not in errors: errors.append("Adjacency Violation")
    return len(errors) == 0, errors

def symbolic_masker(placed_coords, grid_data, rules=['row', 'col', 'color', 'adjacency']):
    placed_queens, true_queens = list(placed_coords), list(placed_coords)    
    for q1 in placed_queens:
        x1, y1 = q1
        color1 = int(grid_data[y1-1, x1-1])
        conflict = False
        for q2 in true_queens:
            if q1 == q2: continue
            x2, y2 = q2
            color2 = int(grid_data[y2-1, x2-1])
            if 'row' in rules and y1 == y2: conflict = True; break
            if 'col' in rules and x1 == x2: conflict = True; break
            if 'color' in rules and color1 == color2: conflict = True; break
            if 'adjacency' in rules and abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1: conflict = True; break
        if conflict and q1 in true_queens:
            true_queens.remove(q1)  
    return set(placed_queens) - set(true_queens)

def plot_error_analysis(neural_stats, final_stats, filename='error_analysis.png'):
    all_errors = sorted(list(set(neural_stats.keys()).union(set(final_stats.keys()))))
    if not all_errors: return

    neural_counts = [neural_stats.get(e, 0) for e in all_errors]
    final_counts = [final_stats.get(e, 0) for e in all_errors]

    x, width = np.arange(len(all_errors)), 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, neural_counts, width, label='Before Symbolic Pass', color='skyblue', edgecolor='black')
    rects2 = ax.bar(x + width/2, final_counts, width, label='After Symbolic Pass', color='salmon', edgecolor='black')

    ax.set_ylabel('Total Number of Violations', fontweight='bold')
    ax.set_title('Error Analysis: Before vs After Symbolic Pass', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(all_errors, rotation=25, ha="right", fontsize=10)
    ax.legend()
    ax.bar_label(rects1, padding=3); ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_pipeline(args, solver, masker, active_rules):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"Neural_Perfect": 0, "Final_Perfect": 0}
    neural_errs, final_errs = {}, {}

    for trial in range(args.n_trials):
        grid, _, colors = generate_board(args.n_size)
        current_clues, seen_states = set(), []
        placed_coords = []

        # Neural-Only Loop
        for iteration in range(args.max_iters):
            solver_feats = [[x, y, int(grid[y, x]), 1 if (x+1, y+1) in current_clues else 0] for x in range(args.n_size) for y in range(args.n_size)]
            s_tensor = torch.tensor(solver_feats, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad(): probs = torch.sigmoid(solver(s_tensor)).squeeze(0).cpu().numpy()
            
            predictions = (probs > args.prob_solver).astype(int)
            placed_coords = [(x+1, y+1) for i, (x, y, _, _) in enumerate(solver_feats) if predictions[i] == 1]
            
            state_id = frozenset(placed_coords)
            if state_id in seen_states and len(current_clues) < args.n_size:
                best_new_idx = np.argmax([p if (solver_feats[i][0]+1, solver_feats[i][1]+1) not in current_clues else -1.0 for i, p in enumerate(probs)])
                current_clues.add((solver_feats[best_new_idx][0]+1, solver_feats[best_new_idx][1]+1))
                continue
            seen_states.append(state_id)

            m_feats = [[f[0], f[1], f[2], predictions[i]] for i, f in enumerate(solver_feats)]
            m_tensor = torch.tensor(m_feats, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad(): m_probs = torch.sigmoid(masker(m_tensor)).squeeze(0).cpu().numpy()
            
            m_decisions = ((m_probs > args.prob_mask) & (predictions == 1)).astype(int)
            m_decisions_neg = ((m_probs > args.prob_mask) & (predictions == 0)).astype(int)
            
            new_clues = set()
            for i, (x, y, _, has_q) in enumerate(m_feats):
                if has_q == 1 and m_decisions[i] == 0: new_clues.add((x+1, y+1))
                if has_q == 0 and m_decisions_neg[i] == 1: new_clues.add((x+1, y+1))
            
            if new_clues == current_clues and iteration > 0: break
            current_clues = new_clues

        is_neural_valid, n_errors = check_symbolic(placed_coords, grid, args.n_size)
        if is_neural_valid: results["Neural_Perfect"] += 1
        else:
            for e in n_errors: neural_errs[e] = neural_errs.get(e, 0) + 1

        # Symbolic-Masker Check & Final Pass
        flagged = symbolic_masker(placed_coords, grid, rules=active_rules)
        final_clues = {q for q in placed_coords if q not in flagged}
        
        f_feats = [[x, y, int(grid[y, x]), 1 if (x+1, y+1) in final_clues else 0] for x in range(args.n_size) for y in range(args.n_size)]
        f_tensor = torch.tensor(f_feats, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad(): f_probs = torch.sigmoid(solver(f_tensor)).squeeze(0).cpu().numpy()
        
        f_preds = (f_probs > args.prob_solver).astype(int)
        final_coords = [(x+1, y+1) for i, (x, y, _, _) in enumerate(f_feats) if f_preds[i] == 1]

        is_final_valid, f_errors = check_symbolic(final_coords, grid, args.n_size)
        if is_final_valid: results["Final_Perfect"] += 1
        else:
            for e in f_errors: final_errs[e] = final_errs.get(e, 0) + 1

    acc_n = (results["Neural_Perfect"] / args.n_trials) * 100
    acc_f = (results["Final_Perfect"] / args.n_trials) * 100
    print(f"[{'+'.join(active_rules) if active_rules else 'Baseline'}] Neural: {acc_n:.2f}% | Final: {acc_f:.2f}% | Boost: +{acc_f-acc_n:.2f}%")
    return acc_f, neural_errs, final_errs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_size", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--n_trials", type=int, default=5000) 
    parser.add_argument("--max_iters", type=int, default=4)
    parser.add_argument("--prob_solver", type=float, default=0.9)
    parser.add_argument("--prob_mask", type=float, default=0.85)
    parser.add_argument("--solver_path", type=str, default="queens_model.pth")
    parser.add_argument("--mask_path", type=str, default="alternative_mask_predictor.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        solver = QueensTransformer(args.n_size, args.d_model, args.n_heads, args.num_layers).to(device)
        solver.load_state_dict(torch.load(args.solver_path, map_location=device, weights_only=True))
        
        masker = AlternativeMaskTransformer(args.n_size, args.d_model, args.n_heads, args.num_layers).to(device)
        masker.load_state_dict(torch.load(args.mask_path, map_location=device, weights_only=True))
        
        os.makedirs("error_analysis_graphs", exist_ok=True)
        rules = ['color', 'col', 'row', 'adjacency']
        
        for r in range(len(rules) + 1):
            for combo in combinations(rules, r):
                active = list(combo)
                set_seed(6700) 
                _, n_errs, f_errs = evaluate_pipeline(args, solver, masker, active)
                
                rule_str = "_".join(active) if active else "no_rules"
                plot_error_analysis(n_errs, f_errs, f"error_analysis_graphs/graph_{rule_str}.png")
                
    except FileNotFoundError as e:
        print(f"Ensure you have trained both models before running the pipeline!\nError: {e}")