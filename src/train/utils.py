
import torch, numpy as np

def resample_time(x, T_target):
    """x: [T, J, C] -> resampled to T_target using simple linear interpolation over time."""
    T, J, C = x.shape
    if T == T_target:
        return x
    # indices in original time
    src = np.linspace(0, T-1, num=T_target)
    x_new = np.empty((T_target, J, C), dtype=x.dtype)
    for j in range(J):
        for c in range(C):
            x_new[:, j, c] = np.interp(src, np.arange(T), x[:, j, c])
    return x_new

def collate_and_resample(items, T_target):
    """Returns tensors:
       pose: [B, T_target, J, C], text(list), gloss(list), meta(list)
    """
    X = []
    texts, glosses, metas = [], [], []
    for it in items:
        arr = it["pose"].numpy() if isinstance(it["pose"], torch.Tensor) else it["pose"]
        arr = resample_time(arr, T_target)
        X.append(torch.from_numpy(arr).float())
        texts.append(it.get("text"))
        glosses.append(it.get("gloss"))
        metas.append(it["meta"])
    X = torch.stack(X, dim=0)  # [B, T, J, C]
    return {"pose": X, "text": texts, "gloss": glosses, "meta": metas}

def flatten_TJC(x):
    # x: [B, T, J, C] -> [B, T*J*C]
    B, T, J, C = x.shape
    return x.view(B, T*J*C), (T, J, C)
