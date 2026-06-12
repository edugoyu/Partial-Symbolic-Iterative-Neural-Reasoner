import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

from utils.common import set_seed, generate_board, CustomEncoderLayer

def corrupt_board_with_tracking(valid_coords, n, min_c, max_c):
    coords = list(valid_coords)
    modified_set = set()
    num_corruptions = random.randint(min_c, max_c) 
    
    for _ in range(num_corruptions):
        action = random.choice(['add', 'move', 'remove'])
        if action == 'add':
            x, y = random.randint(1, n), random.randint(1, n)
            if (x, y) not in coords:
                coords.append((x, y))
                modified_set.add((x, y))
        elif action == 'move' and coords:
            idx = random.randint(0, len(coords) - 1)
            modified_set.add(coords.pop(idx))
            x, y = random.randint(1, n), random.randint(1, n)
            if (x, y) not in coords:
                coords.append((x, y))
                modified_set.add((x, y))
        elif action == 'remove' and coords:
            idx = random.randint(0, len(coords) - 1)
            modified_set.add(coords.pop(idx))
                
    return coords, modified_set

class AlternatingMaskDataset(Dataset):
    def __init__(self, n, num_samples, min_c, max_c):
        self.samples = []
        for _ in range(num_samples):
            grid, valid_coords, _ = generate_board(n)
            bad_coords, modified_set = corrupt_board_with_tracking(valid_coords, n, min_c, max_c)
            queen_set = set(bad_coords)
            
            symbolic_grid = []
            for x in range(1, n + 1):
                for y in range(1, n + 1):
                    color_idx = int(grid[y-1, x-1])
                    has_queen = 1 if (x, y) in queen_set else 0
                    is_modified = 1 if (x, y) in modified_set else 0 
                    symbolic_grid.append([x-1, y-1, color_idx, has_queen, is_modified])
            
            data_tensor = torch.tensor(symbolic_grid, dtype=torch.long)
            self.samples.append((data_tensor[:, :4], data_tensor[:, 4].float()))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

class AlternativeMaskTransformer(nn.Module):
    def __init__(self, n, d_model, nhead, num_layers):
        super().__init__()
        self.x_embed = nn.Embedding(n, d_model)
        self.y_embed = nn.Embedding(n, d_model)
        self.color_embed = nn.Embedding(n, d_model)
        self.state_embed = nn.Embedding(2, d_model) 
        self.layers = nn.ModuleList([CustomEncoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, features):
        x = self.x_embed(features[:,:,0]) + self.y_embed(features[:,:,1]) + \
            self.color_embed(features[:,:,2]) + self.state_embed(features[:,:,3])
        for layer in self.layers:
            x, _ = layer(x)
        return self.output_head(x).squeeze(-1)

def train_alternating_masker(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Mask-Predictor on {device}...")
    
    train_loader = DataLoader(AlternatingMaskDataset(args.n_size, args.train_samples, args.min_c, args.max_c), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(AlternatingMaskDataset(args.n_size, args.val_samples, args.min_c, args.max_c), batch_size=args.batch_size)

    model = AlternativeMaskTransformer(args.n_size, args.d_model, args.n_heads, args.num_layers).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.error_penalty]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for feat, target in train_loader:
            feat, target = feat.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feat), target)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d} complete | Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_size", type=int, default=10)
    parser.add_argument("--train_samples", type=int, default=15000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--min_c", type=int, default=1)
    parser.add_argument("--max_c", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--error_penalty", type=float, default=10.0)
    parser.add_argument("--save_path", type=str, default="alternative_mask_predictor.pth")
    
    args = parser.parse_args()
    set_seed(6723843)
    train_alternating_masker(args)