import torch
import numpy as np
from collections import OrderedDict, deque, namedtuple
import random
from torch.utils.data import DataLoader

def build_loader(full_ds, subset_idx, shuffle_flag, batch_size):
    subset = torch.utils.data.Subset(full_ds, subset_idx)
    return DataLoader(subset,
                      batch_size=batch_size,
                      shuffle=shuffle_flag,
                      drop_last=False)

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
    def __len__(self):
        return len(self.xs)
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], idx


CacheEntryW = namedtuple(
    "CacheEntry",
    ["y_pred", "x", "weight", "z_star", "info", "elbo"]
)

CacheEntryW2 = namedtuple(
    "CacheEntry",
    ["x", "y", "weight", "z_star", "elbo"]
)

CacheEntry = namedtuple(
    "CacheEntry",
    ["x", "y", "elbo"]
)

class ReplayBufferList:
    def __init__(self, capacity: int, device: torch.device | str):
        self.capacity: int = int(capacity)
        self.device = device
        self._buffer: OrderedDict[int, list[CacheEntry]] = OrderedDict()
        self._order: deque[tuple[int, int]] = deque()

    @property
    def size(self) -> int:
        return len(self._order)

    def _evict_if_needed(self):
        """Pop the oldest entries until total size <= capacity."""
        while self.size > self.capacity:
            old_k, old_pos = self._order.popleft()
            buf_list = self._buffer.get(old_k)
            if buf_list is None:
                continue

            buf_list[old_pos] = None
            while buf_list and buf_list[0] is None:
                buf_list.pop(0)

            if not buf_list:
                del self._buffer[old_k]

    @torch.no_grad()
    def put(self,
            idx_tensor: torch.Tensor,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            weight: torch.Tensor,
            z_star: torch.Tensor,
            elbo: torch.Tensor):
        for i, k in enumerate(idx_tensor.tolist()):
            k = int(k)
            entry = CacheEntryW2(
                x[i].cpu(),
                y_pred[i].cpu(),
                weight[i].cpu(),
                z_star[i].cpu(),
                elbo[i].cpu()
            )
            self._buffer.setdefault(k, []).append(entry)
            self._order.append((k, len(self._buffer[k]) - 1))

        self._evict_if_needed()

    def get(self, idx: int, *, latest_only: bool = True):
        lst = self._buffer.get(int(idx))
        if lst is None:
            return None
        self._buffer.move_to_end(int(idx))
        return lst[-1] if latest_only else lst

    def get_many(self, idx_list, *, latest_only: bool = True):
        xs, ys, weights, z_stars, elbos = [], [], [], [], []

        for k in idx_list:
            entries = self._buffer.get(int(k))
            if not entries:
                continue
            self._buffer.move_to_end(int(k))

            chosen = entries[-1] if latest_only else entries
            chosen = [chosen] if isinstance(chosen, CacheEntry) else chosen
            for e in chosen:
                xs.append(e.x)
                ys.append(e.y)
                weights.append(e.weight)
                z_stars.append(e.z_star)
                elbos.append(e.elbo)

        if not xs:
            return None, None, None, None, None

        x_ls = torch.stack(xs, dim=0).to(self.device)
        y_ls = torch.stack(ys, dim=0).to(self.device)
        weight_ls = torch.stack(weights, dim=0).to(self.device)
        z_star_ls = torch.stack(z_stars, dim=0).to(self.device)
        elbo_ls = torch.stack(elbos, dim=0).to(self.device)
        return x_ls, y_ls, weight_ls, z_star_ls, elbo_ls

    def sample(self, k: int):
        k = min(k, self.size)
        sampled_refs = random.sample(list(self._order), k)  # (k, pos)
        ids, batch = [], []
        for idx, pos in sampled_refs:
            ids.append(idx)
            batch.append(self._buffer[idx][pos])
        return ids, batch

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self._buffer = OrderedDict()
    
    @torch.no_grad()
    def put(self, idx_tensor, x_rep_bmc, y_rep_bmc, elbo_bmc):
        for i, k in enumerate(idx_tensor):
            k = int(k)
            if k in self._buffer:
                self._buffer.move_to_end(k)
            self._buffer[k] = CacheEntry(
                x_rep_bmc[i].cpu(),                  # (mc, x_dim)
                y_rep_bmc[i].cpu(),                   # (mc, y_dim)
                elbo_bmc[i].cpu()                   # (mc, y_dim)
            )
            if len(self._buffer) > self.capacity:
                self._buffer.popitem(last=False)
    
    def get(self, idx: int):
        entry = self._buffer.get(int(idx))
        if entry is not None:
            self._buffer.move_to_end(int(idx))
        return entry

    def get_many(self, idx_list):
        entries = []
        for k in idx_list:
            k_int = int(k)
            e = self._buffer.get(k_int)
            if e is not None:
                self._buffer.move_to_end(k_int)
            entries.append(e)
        if len([e.x for e in entries if e is not None]) == 0:
            return None, None, None
        x_ls = torch.stack([e.x for e in entries if e is not None], dim=0).to(self.device)
        y_ls = torch.stack([e.y for e in entries if e is not None], dim=0).to(self.device)
        elbo_ls = torch.stack([e.elbo for e in entries if e is not None], dim=0).to(self.device)
        return y_ls, x_ls, elbo_ls

    def sample(self, k):
        ids = random.sample(list(self._buffer.keys()), k)
        batch = [self._buffer[i] for i in ids]          # list[CacheEntry]
        return ids, batch

class ReplayBufferW:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self._buffer = OrderedDict()
    
    def get_many(self, idx_tensor):
        y_ls, x_ls, w_ls, z_ls, info_ls, elbo_ls = [], [], [], [], [], []

        for t in idx_tensor:
            entry = self.get(int(t))
            if entry is None:
                return None
            y_ls.append(entry.y_pred)
            x_ls.append(entry.x)
            w_ls.append(entry.weight)
            z_ls.append(entry.z_star)
            info_ls.append(torch.tensor(entry.info))
            elbo_ls.append(entry.elbo)

        y_batch = torch.stack(y_ls, dim=0).to(self.device) # (64, 20, 24)
        x_batch = torch.stack(x_ls, dim=0).to(self.device) # (64, 149)
        w_batch = torch.stack(w_ls, dim=1).to(self.device) # (20, 64)
        z_batch = torch.stack(z_ls, dim=0).to(self.device) # (64, 24)
        info_batch = torch.stack(info_ls, dim=0).to(self.device) # (64, 70, 70)
        elbo_batch = torch.stack(elbo_ls, dim=1).to(self.device) # (20, 64)

        return y_batch, x_batch, w_batch, z_batch, info_batch, elbo_batch

    def put(self, idx_list, **fields):
        for i, t in enumerate(idx_list):
            key = int(t)
            self.put_single(
                key,
                y_pred = fields["y_pred"][i], # (1240, 24)
                x      = fields["x"][i],  #(1240, 149)
                weight = fields["weight"][:, i], # (20, 64)
                z_star = fields["z_star"][i], # (64, 24)
                info   = fields["info"]['KKT_matrices'][i], # (64, 70, 70)
                # info   = fields["info"]['inv_matrices'][i], # (64, 70, 70)
                elbo   = fields["elbo"][:, i], # (20, 64)
            )
    
    def get(self, idx):
        entry = self._buffer.get(idx)
        if entry is not None:
            self._buffer.move_to_end(idx)
        return entry

    def put_single(self, idx, *, y_pred, x, weight, z_star, info, elbo):
        if idx in self._buffer:
            self._buffer.move_to_end(idx)
        self._buffer[idx] = CacheEntryW(
            y_pred.detach().cpu(),
            x.detach().cpu(),
            weight.detach().cpu(),
            z_star.detach().cpu(),
            info,
            elbo.detach().cpu()
        )
        if len(self._buffer) > self.capacity:
            self._buffer.popitem(last=False)
            print("POP")

    def sample(self, k):
        ids = random.sample(list(self._buffer.keys()), k)
        batch = [self._buffer[i] for i in ids]
        return batch

def print_header(msg):
    print('===>', msg)


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().numpy()


def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def get_sizes(G, A=None):
    if G.dim() == 2:
        nineq, nz = G.size()
        nBatch = 1
    elif G.dim() == 3:
        nBatch, nineq, nz = G.size()
    if A is not None:
        neq = A.size(1) if A.nelement() > 0 else 0
    else:
        neq = None
    # nBatch = batchedTensor.size(0) if batchedTensor is not None else None
    return nineq, nz, neq, nBatch


def bdiag(d):
    nBatch, sz = d.size()
    D = torch.zeros(nBatch, sz, sz).type_as(d)
    I = torch.eye(sz).repeat(nBatch, 1, 1).type_as(d).bool()
    D[I] = d.squeeze().view(-1)
    return D


def expandParam(X, nBatch, nDim):
    if X.ndimension() in (0, nDim) or X.nelement() == 0:
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([nBatch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")


def extract_nBatch(Q, p, G, h, A, b):
    dims = [3, 2, 3, 2, 3, 2]
    params = [Q, p, G, h, A, b]
    for param, dim in zip(params, dims):
        if param.ndimension() == dim:
            return param.size(0)
    return 1
