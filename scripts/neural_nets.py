import os
import argparse
import math
import random
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

class ScoringModel(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_emb: torch.Tensor, cand_emb: torch.Tensor):
        # user_emb: (B, D)
        # cand_emb: (B, C, D)
        B, C, D = cand_emb.shape
        user_expand = user_emb.unsqueeze(1).expand(-1, C, -1)  # (B,C,D)
        x = torch.cat([user_expand, cand_emb], dim=-1).view(B * C, 2 * D)  # (B*C, 2D)
        logits = self.mlp(x).view(B, C)  # (B, C)
        return logits

 
class MINDBehaviorDataset(Dataset):
    """
    Reads behaviors.tsv lines:
    userId \t time \t history(space-separated news ids) \t impressions(space-separated newsid-0/1)
    Produces items: (history_ids: List[str], cand_ids: List[str], labels: List[int])
    """
    
    def __init__(self, behaviors_path: str, news_in_vocab: set, max_history: int = 50, neg_sample_size: int = 4, sample_negatives: bool = True):
        self.samples = []
        self.max_history = max_history
        self.neg_sample_size = neg_sample_size
        self.sample_negatives = sample_negatives

        if not os.path.exists(behaviors_path):
            raise FileNotFoundError(f"behaviors file not found: {behaviors_path}")

        # Track statistics for diagnostic messages
        total_lines = 0
        skipped_lines = 0
        
        # Try using pandas first for more robust handling
        try:
            df = pd.read_csv(behaviors_path, sep='\t', header=None, quoting=3, engine='python', dtype=str).fillna('')
            print(f"Loaded behaviors file with shape: {df.shape}")
            
            # Handle different column layouts
            if df.shape[1] >= 5:  # Standard MIND format
                history_col = 3
                impressions_col = 4
            else:
                # Assume history is second-to-last and impressions is last column
                history_col = df.shape[1] - 2 if df.shape[1] > 2 else 0
                impressions_col = df.shape[1] - 1
                
            print(f"Using columns: history={history_col}, impressions={impressions_col}")
            
            for _, row in df.iterrows():
                total_lines += 1
                
                # Extract history and impressions
                history = str(row[history_col]) if history_col < len(row) else ""
                impressions = str(row[impressions_col]) if impressions_col < len(row) else ""
                
                history_ids = [h.strip() for h in history.split() if h.strip() and h.strip() in news_in_vocab]
                
                # Handle different impression formats
                cand_ids = []
                labels = []
                
                # Try splitting by spaces first (standard format)
                imp_list = impressions.split()
                for imp in imp_list:
                    if '-' in imp:
                        try:
                            nid, lab = imp.rsplit('-', 1)
                            nid = nid.strip()
                            if nid not in news_in_vocab:
                                continue
                            lab_int = int(lab)
                            cand_ids.append(nid)
                            labels.append(lab_int)
                        except Exception:
                            continue
                
                # If no candidates found, try alternative formats
                if not cand_ids and impressions:
                    # Try comma-separated format
                    for imp in impressions.split(','):
                        if '-' in imp:
                            try:
                                nid, lab = imp.rsplit('-', 1)
                                nid = nid.strip()
                                if nid not in news_in_vocab:
                                    continue
                                lab_int = int(lab)
                                cand_ids.append(nid)
                                labels.append(lab_int)
                            except Exception:
                                continue
                
                if cand_ids:
                    self.samples.append((history_ids, cand_ids, labels))
                else:
                    skipped_lines += 1
                    
        except Exception as e:
            print(f"Pandas loading failed: {e}. Falling back to manual line parsing.")
            
            # Fall back to manual line parsing
            with open(behaviors_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    total_lines += 1
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 3:
                        skipped_lines += 1
                        continue
                    
                    # Extract history and impressions - handle different formats
                    if len(parts) >= 5:  # Standard MIND format
                        history = parts[3]
                        impressions = parts[4]
                    else:
                        # Try to be robust with column positions
                        history = parts[-2] if len(parts) >= 3 else ""
                        impressions = parts[-1]
                    
                    history_ids = [h.strip() for h in history.split() if h.strip() and h.strip() in news_in_vocab]
                    
                    cand_ids = []
                    labels = []
                    
                    # Process impressions (try different formats)
                    for imp in impressions.split():
                        if '-' in imp:
                            try:
                                nid, lab = imp.rsplit('-', 1)
                                nid = nid.strip()
                                if nid not in news_in_vocab:
                                    continue
                                lab_int = int(lab)
                                cand_ids.append(nid)
                                labels.append(lab_int)
                            except Exception:
                                continue
                    
                    if cand_ids:
                        self.samples.append((history_ids, cand_ids, labels))
                    else:
                        skipped_lines += 1

        # More detailed diagnostic output
        print(f"Loaded {len(self.samples)} valid samples from {behaviors_path}")
        print(f"Total lines: {total_lines}, Skipped: {skipped_lines}")
        if total_lines > 0:
            print(f"Retention rate: {len(self.samples)/total_lines:.2%}")
            
        # Sample debugging - show first few samples
        if self.samples:
            print("First sample:")
            hist, cands, labs = self.samples[0]
            print(f"  History: {hist[:5]}{'...' if len(hist) > 5 else ''}")
            print(f"  Candidates: {cands[:5]}{'...' if len(cands) > 5 else ''}")
            print(f"  Labels: {labs[:5]}{'...' if len(labs) > 5 else ''}")
        else:
            print("WARNING: No valid samples found in behaviors file!")
            # Let's print out some raw lines to help diagnose the issue
            try:
                with open(behaviors_path, 'r', encoding='utf-8') as f:
                    sample_lines = [next(f).strip() for _ in range(3) if f]
                    print("Sample raw lines:")
                    for i, line in enumerate(sample_lines):
                        print(f"  Line {i+1}: {line[:100]}{'...' if len(line) > 100 else ''}")
            except Exception as e:
                print(f"Could not read sample lines: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history_ids, cand_ids, labels = self.samples[idx]
        # truncate history
        history_ids = history_ids[-self.max_history:]
        # optionally sample negatives to reduce candidate set size
        if self.sample_negatives:
            pos_idx = [i for i, l in enumerate(labels) if l == 1]
            neg_idx = [i for i, l in enumerate(labels) if l == 0]
            if len(pos_idx) == 0:
                if len(neg_idx) == 0:
                    chosen = list(range(len(labels)))
                else:
                    chosen = random.sample(neg_idx, min(self.neg_sample_size, len(neg_idx)))
            else:
                chosen = []
                for p in pos_idx:
                    chosen.append(p)
                    if neg_idx:
                        chosen += random.sample(neg_idx, min(self.neg_sample_size, len(neg_idx)))
            # unique preserving order
            seen = set()
            final_idx = []
            for i in chosen:
                if i not in seen:
                    seen.add(i)
                    final_idx.append(i)
            cand_final = [cand_ids[i] for i in final_idx]
            labels_final = [labels[i] for i in final_idx]
            return history_ids, cand_final, labels_final
        else:
            return history_ids, cand_ids, labels


def build_news_embeddings(news_map: Dict[str, str], svd_dim: int = 128, max_features: int = 50000):
    """
    Fit TF-IDF on news texts and reduce with TruncatedSVD to produce dense embeddings.
    Returns: news_ids(list), embeddings(np.ndarray shape (N, D)), vectorizer, svd
    """
    if not news_map:
        raise ValueError("Empty news_map provided to build_news_embeddings")
        
    news_ids = list(news_map.keys())
    texts = [news_map[nid] for nid in news_ids]
    
    # Add safeguards for very small datasets
    actual_max_features = min(max_features, len(texts) * 10)
    actual_svd_dim = min(svd_dim, int(len(texts) * 0.8))
    
    print(f"Building TF-IDF with {actual_max_features} features for {len(texts)} documents")
    tfidf = TfidfVectorizer(max_features=actual_max_features, ngram_range=(1,2))
    X = tfidf.fit_transform(texts)  # sparse (N, V)
    
    print(f"TF-IDF shape: {X.shape}, reducing to {actual_svd_dim} dimensions")
    svd = TruncatedSVD(n_components=actual_svd_dim, random_state=42)
    emb = svd.fit_transform(X)  # (N, D)
    
    # Check explained variance
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"SVD explained variance: {explained_var:.2%}")
    
    # normalize embeddings
    emb = normalize(emb, axis=1)
    return news_ids, emb.astype(np.float32), tfidf, svd

def collate_batch(batch, newsid_to_idx: Dict[str, int], news_emb_np: np.ndarray, device: torch.device):
    """
    Convert batch of (history_ids, cand_ids, labels) into tensors:
    user_embs: (B, D)
    cand_embs: (B, C, D)
    labels: (B, C)
    """
    if not batch:
        # Return empty tensors with the right dimensions
        D = news_emb_np.shape[1] if news_emb_np.size > 0 else 0
        return (torch.zeros((0, D), device=device),
                torch.zeros((0, 0, D), device=device),
                torch.zeros((0, 0), device=device))
                
    B = len(batch)
    max_c = max(len(x[1]) for x in batch) if B > 0 else 0
    D = news_emb_np.shape[1]
    user_embs = []
    cand_embs = []
    labels = []
    
    for history_ids, cand_ids, lab in batch:
        # user embedding = mean of history embeddings
        hist_idxs = [newsid_to_idx[n] for n in history_ids if n in newsid_to_idx]
        if hist_idxs:
            u = np.mean(news_emb_np[hist_idxs], axis=0)
        else:
            u = np.zeros(D, dtype=np.float32)
            
        # candidate embeddings and pad
        cand_idxs = [newsid_to_idx[n] for n in cand_ids if n in newsid_to_idx]
        if not cand_idxs:
            # Skip samples with no valid candidates
            continue
            
        cand_e = news_emb_np[cand_idxs]
        lab_filtered = [lab[i] for i in range(len(cand_ids)) if cand_ids[i] in newsid_to_idx]
        
        pad_count = max_c - cand_e.shape[0]
        if pad_count > 0:
            cand_e = np.vstack([cand_e, np.zeros((pad_count, D), dtype=np.float32)])
            lab_filtered = lab_filtered + [0] * pad_count
        user_embs.append(u)
        cand_embs.append(cand_e)
        labels.append(lab_filtered)
    
    if not user_embs:
        # If all samples were filtered out, return empty tensors
        return (torch.zeros((0, D), device=device),
                torch.zeros((0, 0, D), device=device),
                torch.zeros((0, 0), device=device))
                
    user_embs = torch.from_numpy(np.stack(user_embs)).to(device)
    cand_embs = torch.from_numpy(np.stack(cand_embs)).to(device)
    labels = torch.from_numpy(np.array(labels, dtype=np.float32)).to(device)
    return user_embs, cand_embs, labels

 
def hr_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 5):
    idx = np.argsort(-scores)[:k]
    return 1.0 if labels[idx].sum() > 0 else 0.0

def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 5):
    idx = np.argsort(-scores)[:k]
    gains = labels[idx]
    if gains.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k+2))
    return float((gains * discounts).sum() / discounts[0])

def evaluate(model: nn.Module, dataloader: DataLoader, newsid_to_idx: Dict[str,int], news_emb_np: np.ndarray, device: torch.device, k: int = 5):
    model.eval()
    HRs = []
    NDCGs = []
    with torch.no_grad():
        for batch in dataloader:
            user_embs, cand_embs, labels = collate_batch(batch, newsid_to_idx, news_emb_np, device)
            if user_embs.shape[0] == 0:
                continue
            logits = model(user_embs, cand_embs)  # (B, C)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()
            for i in range(probs.shape[0]):
                HRs.append(hr_at_k(probs[i], labels_np[i], k))
                NDCGs.append(ndcg_at_k(probs[i], labels_np[i], k))
    return float(np.mean(HRs)) if HRs else 0.0, float(np.mean(NDCGs)) if NDCGs else 0.0

 
def online_update(model: nn.Module, optimizer: optim.Optimizer, loss_fn, user_emb_batch: torch.Tensor, cand_emb_batch: torch.Tensor, labels_batch: torch.Tensor, device: torch.device):
    model.train()
    user_emb_batch = user_emb_batch.to(device)
    cand_emb_batch = cand_emb_batch.to(device)
    labels_batch = labels_batch.to(device)
    optimizer.zero_grad()
    logits = model(user_emb_batch, cand_emb_batch)
    loss = loss_fn(logits, labels_batch)
    loss.backward()
    optimizer.step()
    return float(loss.item())

# -----------------------
# Main training flow
# -----------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
    print("Using device:", device)

    # resolve paths relative to repo root (one level up from this script)
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    def resolve(p):
        if not p:
            return p
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(repo_root, p))

    # 1) Load news texts
    news_path = resolve(args.news or "Data/MINDsmall_train/news.tsv")
    print("Loading news from", news_path)
    if args.cache_embeddings and os.path.exists(os.path.join(resolve(args.cache_embeddings), "news_emb.npy")):
        # load cached embeddings if present
        try:
            cache_dir = resolve(args.cache_embeddings)
            news_ids = list(np.load(os.path.join(cache_dir, "news_ids.npy"), allow_pickle=True).astype(str))
            news_emb_np = np.load(os.path.join(cache_dir, "news_emb.npy"), allow_pickle=True).astype(np.float32)
            print(f"Loaded cached news embeddings from {cache_dir} - {len(news_ids)} embeddings of dim {news_emb_np.shape[1]}")
        except Exception as e:
            print(f"Error loading cached embeddings: {str(e)}. Will rebuild from source.")
            news_ids, news_emb_np = None, None
    else:
        news_ids, news_emb_np = None, None
        
    if news_ids is None or news_emb_np is None:
        if not os.path.exists(news_path):
            raise FileNotFoundError(f"news file not found: {news_path}")
        news_map = {}
        with open(news_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) < 2:  # At minimum need ID and some text
                    continue
                nid = parts[0]
                title = parts[3] if len(parts) > 3 else ""
                abstract = parts[4] if len(parts) > 4 else ""
                text = (title + " " + abstract).strip()
                if not text:
                    text = title or abstract or "unknown_content"  # Non-empty placeholder
                news_map[nid] = text
                
        if not news_map:
            raise ValueError(f"No valid news items found in {news_path}. Check file format.")
            
        # 2) Build news embeddings (TF-IDF + SVD)
        print(f"Building news embeddings (TF-IDF + SVD) for {len(news_map)} items...")
        news_ids, news_emb_np, tfidf, svd = build_news_embeddings(
            news_map, 
            svd_dim=min(args.emb_dim, len(news_map) - 1),  # Ensure SVD dim is valid
            max_features=args.max_features
        )
        if args.cache_embeddings:
            try:
                cache_dir = resolve(args.cache_embeddings)
                os.makedirs(cache_dir, exist_ok=True)
                np.save(os.path.join(cache_dir, "news_ids.npy"), np.array(news_ids))
                np.save(os.path.join(cache_dir, "news_emb.npy"), news_emb_np)
                print("Saved news embeddings to", cache_dir)
            except Exception as e:
                print(f"Warning: Failed to save embeddings cache: {str(e)}")

    print(f"Built embeddings for {len(news_ids)} news items, emb_dim={news_emb_np.shape[1]}")

    # 3) Datasets - DIAGNOSTIC + SAFE CHECK
    behaviors_train = resolve(args.behaviors_train or "Data/MINDsmall_train/behaviors.tsv")
    behaviors_dev = resolve(args.behaviors_dev or "Data/MINDsmall_dev/behaviors.tsv")

    if not os.path.exists(behaviors_train):
        raise FileNotFoundError(f"behaviors_train file not found: {behaviors_train}")
    if not os.path.exists(behaviors_dev):
        raise FileNotFoundError(f"behaviors_dev file not found: {behaviors_dev}")

    # Let's peek at the file structure to understand format
    print("\nInspecting behaviors file structure...")
    try:
        with open(behaviors_train, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"First line: {first_line}")
            fields = first_line.split('\t')
            print(f"Number of tab-separated fields: {len(fields)}")
            if len(fields) >= 4:
                print(f"Impressions field (expected in position 4): {fields[3]}")
                # Check if impressions are in expected format
                if fields[3] and '-' in fields[3]:
                    print("Format looks correct with impressions containing ID-label pairs")
                else:
                    # Maybe impressions are in a different column?
                    for i, field in enumerate(fields):
                        if field and '-' in field and any(c.isdigit() for c in field):
                            print(f"Possible impressions field found at position {i}: {field}")
    except Exception as e:
        print(f"Error analyzing file structure: {e}")

    def collect_referenced_ids(path):
        refs = set()
        sample_lines = []
        
        # Try pandas first - more robust to various file formats
        try:
            df = pd.read_csv(path, sep='\t', header=None, quoting=3, engine='python', dtype=str).fillna('')
            print(f"Pandas loaded shape: {df.shape}")
            
            # Determine impressions column - usually last column
            imp_col = df.shape[1] - 1
            
            # Sample a few rows for debugging
            for i in range(min(5, len(df))):
                if i < len(df):
                    sample_lines.append(str(dict(df.iloc[i])))
            
            # Extract news IDs from impressions
            for imp in df[imp_col]:
                for item in str(imp).split():
                    if '-' in item:
                        try:
                            nid, _ = item.rsplit('-', 1)
                            refs.add(nid.strip())
                        except:
                            pass
                    else:
                        # If no label separator, assume it's just an ID
                        nid = item.strip()
                        if nid:
                            refs.add(nid)
                        
            print(f"Collected {len(refs)} unique referenced news ids from pandas")
        except Exception as e:
            print(f"Pandas parsing failed: {e}")
            refs = set()
        
        # If pandas failed or found nothing, try direct file reading
        if len(refs) == 0:
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    for i, line in enumerate(fh):
                        if i < 10:
                            sample_lines.append(line.rstrip('\n'))
                        
                        parts = line.strip().split('\t')
                        # Try the last column as impressions
                        impressions = parts[-1] if parts else ""
                        
                        for imp in str(impressions).split():
                            if '-' in imp:
                                try:
                                    nid, _ = imp.rsplit('-', 1)
                                    refs.add(nid.strip())
                                except:
                                    pass
                
                print(f"Collected {len(refs)} unique referenced news ids from direct file reading")
            except Exception as e:
                print(f"Direct file reading failed: {e}")
    
        # Log sample lines for debugging
        print(f"--- Sample data from {path} ---")
        for i, s in enumerate(sample_lines[:3]):
            print(f"Sample {i+1}: {s[:200]}...")
        
        return refs

    train_refs = collect_referenced_ids(behaviors_train)
    dev_refs = collect_referenced_ids(behaviors_dev)
    inter_train = train_refs & set(news_ids)
    inter_dev = dev_refs & set(news_ids)
    print(f"behaviors_train references {len(train_refs)} unique news ids, {len(inter_train)} present in news set")
    print(f"behaviors_dev references {len(dev_refs)} unique news ids, {len(inter_dev)} present in news set")

    if len(inter_train) == 0:
        raise RuntimeError(
            "No overlap between behaviors_train impressions and news IDs.\n"
            "Check that you used the correct news.tsv and behaviors.tsv files and that news IDs match.\n"
            "You can inspect samples with: head -n 5 " + behaviors_train.replace('\\','/') + "\n"
        )

    # choose whether to filter to only news referenced in behaviors (helps if news.tsv includes extra items)
    if args.filter_missing_news:
        common_news_ids = list(inter_train | inter_dev)
        if len(common_news_ids) == 0:
            raise RuntimeError("No common news IDs after filtering; aborting.")
        idxs = [news_ids.index(nid) for nid in common_news_ids if nid in news_ids]
        news_emb_np = news_emb_np[idxs]
        news_ids = common_news_ids
    else:
        common_news_ids = news_ids

    newsid_to_idx = {nid: i for i, nid in enumerate(news_ids)}

    train_ds = MINDBehaviorDataset(behaviors_train, set(common_news_ids), max_history=args.max_history, neg_sample_size=args.neg_per_pos, sample_negatives=True)
    dev_ds = MINDBehaviorDataset(behaviors_dev, set(common_news_ids), max_history=args.max_history, neg_sample_size=args.neg_per_pos, sample_negatives=False)

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty after filtering. Run the diagnostics printed above to inspect your files and ID formats.")

    # Define a custom collate function that passes each batch element directly to our collate_batch
    def custom_collate(batch):
        return batch

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=custom_collate)

    # 4) Model, optimizer, loss
    model = ScoringModel(args.emb_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # 5) Training loop
    best_ndcg = 0.0
    best_state = None
    
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        total_loss = 0.0
        batches_processed = 0
        empty_batches = 0
        
        for batch in pbar:
            user_embs, cand_embs, labels = collate_batch(batch, newsid_to_idx, news_emb_np, device)
            if user_embs.shape[0] == 0:
                empty_batches += 1
                continue
                
            optimizer.zero_grad()
            logits = model(user_embs, cand_embs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches_processed += 1
            
            if batches_processed % 10 == 0:
                pbar.set_postfix(loss=total_loss / max(1, batches_processed), 
                                 empty=empty_batches)
                                 
        # Evaluate
        hr, ndcg = evaluate(model, dev_loader, newsid_to_idx, news_emb_np, device, k=args.k)
        print(f"Epoch {epoch+1} Eval HR@{args.k}={hr:.4f} NDCG@{args.k}={ndcg:.4f}")
        
        # Save best model
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = model.state_dict().copy()
            print(f"New best model with NDCG@{args.k}={ndcg:.4f}")

    # 6) Save best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model with NDCG@{args.k}={best_ndcg:.4f}")

    # Save PyTorch state and export scoring model to ONNX (fixed candidate count)
    os.makedirs(os.path.dirname(resolve(args.out_model)), exist_ok=True)
    torch.save(model.state_dict(), resolve(args.out_model).replace('.onnx', '.pth'))
    print("Saved PyTorch model to", resolve(args.out_model).replace('.onnx', '.pth'))

    # Export ONNX: requires fixed candidate count for exported model
    try:
        export_cands = args.export_cands
        emb_dim = news_emb_np.shape[1]
        dummy_user = torch.randn(1, emb_dim, device=device)
        dummy_cands = torch.randn(1, export_cands, emb_dim, device=device)
        model.eval()
        torch.onnx.export(model, (dummy_user, dummy_cands), resolve(args.out_model),
                        input_names=["user_emb", "cand_emb"],
                        output_names=["logits"],
                        opset_version=13,
                        dynamic_axes={"user_emb": {0: "batch"}, "cand_emb": {0: "batch"}})
        print("Exported ONNX scoring model to", resolve(args.out_model))
    except Exception as e:
        print(f"Error exporting ONNX model: {str(e)}")
        print("Saved PyTorch model can still be used for inference.")

 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--news", default="Data/MINDsmall_train/news.tsv")
    p.add_argument("--behaviors-train", dest="behaviors_train", default="Data/MINDsmall_train/behaviors.tsv")
    p.add_argument("--behaviors-dev", dest="behaviors_dev", default="Data/MINDsmall_dev/behaviors.tsv")
    p.add_argument("--emb-dim", dest="emb_dim", type=int, default=128)
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--cache-embeddings", type=str, default="", help="dir to save news_ids.npy and news_emb.npy")
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--neg-per-pos", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out-model", default="models/recommender_scoring.onnx")
    p.add_argument("--export-cands", type=int, default=10, help="fixed candidate count for ONNX export")
    p.add_argument("--filter-missing-news", dest="filter_missing_news", action="store_true", help="filter news to only those referenced by behaviors")
    p.add_argument("--cpu-only", action="store_true", help="force CPU usage even if CUDA is available")
    args = p.parse_args()
    
    try:
        main(args)
    except Exception as e:
        import traceback
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        exit(1)