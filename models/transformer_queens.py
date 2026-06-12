import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random

from utils.common import set_seed, generate_board, CustomEncoderLayer

def transform_to_symbolic(n, grid_data, coordinates, num_clues):
    symbolic_grid = []
    queen_set = set(coordinates)
    clue_list = random.sample(coordinates, min(num_clues, n))
    clue_set = set(clue_list)

    for x in range(1, n + 1):
        for y in range(1, n + 1):
            color_idx = int(grid_data[y-1, x-1])
            has_queen = 1 if (x, y) in queen_set else 0
            is_clue = 1 if (x, y) in clue_set else 0
            symbolic_grid.append([x-1, y-1, color_idx, is_clue, has_queen])
            
    return symbolic_grid

class QueensDataset(Dataset):
    def __init__(self, n, num_samples, min_clues, max_clues):
        self.n = n
        self.samples = []
        for _ in range(num_samples):
            grid, coords, _ = generate_board(n)
            num_clues = random.randint(min_clues, max_clues)
            sym_data = transform_to_symbolic(n, grid, coords, num_clues)
            
            data_tensor = torch.tensor(sym_data, dtype=torch.long)
            features = data_tensor[:, :4] 
            targets = data_tensor[:, 4].float() 
            self.samples.append((features, targets))
            
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

class QueensTransformer(nn.Module):
    def __init__(self, n, d_model, nhead, num_layers):
        super().__init__()
        self.x_embed = nn.Embedding(n, d_model)
        self.y_embed = nn.Embedding(n, d_model)
        self.color_embed = nn.Embedding(n, d_model)
        self.clue_embed = nn.Embedding(2, d_model) 
        
        self.layers = nn.ModuleList([CustomEncoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, features, return_attention=False):
        x = self.x_embed(features[:, :, 0]) + self.y_embed(features[:, :, 1]) + \
            self.color_embed(features[:, :, 2]) + self.clue_embed(features[:, :, 3])
        
        attention_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            if return_attention: attention_maps.append(attn_weights)
                
        logits = self.output_head(x).squeeze(-1)
        if return_attention: return logits, attention_maps
        return logits

def train_solver(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Neuro-Solver on: {device}")
    
    train_dataset = QueensDataset(args.n_size, args.train_samples, args.min_clues, args.max_clues)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = QueensDataset(args.n_size, args.val_samples, args.min_clues, args.max_clues)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = QueensTransformer(args.n_size, args.d_model, args.n_heads, args.num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    weight_ratio = ((args.n_size * args.n_size) - args.n_size) / args.n_size
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_ratio]).to(device))
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for features, targets in train_dataloader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval() 
        val_loss, perfect_boards = 0, 0
        with torch.no_grad():
            for features, targets in val_dataloader:
                features, targets = features.to(device), targets.to(device)
                logits = model(features)
                val_loss += criterion(logits, targets).item()
                preds = (torch.sigmoid(logits) > args.pred_threshold).float()
                for i in range(preds.size(0)):
                    if torch.equal(preds[i], targets[i]):
                        perfect_boards += 1
                        
        acc = (perfect_boards / len(val_dataset)) * 100
        print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss/len(train_dataloader):.4f} | Val Loss: {val_loss/len(val_dataloader):.4f} | Acc: {acc:.2f}%")

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_size", type=int, default=10)
    parser.add_argument("--train_samples", type=int, default=15000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--min_clues", type=int, default=0)
    parser.add_argument("--max_clues", type=int, default=10)
    parser.add_argument("--pred_threshold", type=float, default=0.9)
    parser.add_argument("--save_path", type=str, default="queens_model.pth")
    
    args = parser.parse_args()
    set_seed(678267)
    train_solver(args)