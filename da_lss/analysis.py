import time

import numpy as np
import torch


def retrospective_groups(batteries, seed=2026):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    result = {}
    for domain in sorted({b.domain for b in batteries}):
        cells = [b for b in batteries if b.domain == domain]
        if len(cells) < 4 or any(b.soh is None or len(b.cycles) < 2 for b in cells):
            raise ValueError("Clustering needs at least four complete labeled trajectories per domain")
        soh, cycle = [], []
        for b in cells:
            progress = (b.cycles - b.cycles[0]) / (b.cycles[-1] - b.cycles[0])
            grid = np.linspace(0, 1, 100)
            soh.append(np.interp(grid, progress, b.soh))
            cycle.append(np.interp(grid, progress, b.cycles))
        channels = [np.array(soh), np.array(cycle)]
        x = np.concatenate([(a - a.mean()) / max(a.std(), 1e-8) for a in channels], axis=1)
        scores, fits = {}, {}
        for k in range(2, min(10, len(cells) - 1) + 1):
            labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
            if len(np.unique(labels)) < 2:
                continue
            scores[str(k)] = float(silhouette_score(x, labels))
            fits[k] = labels
        if not scores:
            raise ValueError("No distinct clusters found")
        selected = int(max(scores, key=scores.get))
        labels = fits[selected]

        rates = [(b.soh[0] - b.soh[-1]) / (b.cycles[-1] - b.cycles[0]) for b in cells]
        order = sorted(np.unique(labels), key=lambda c: np.mean(np.array(rates)[labels == c]), reverse=True)
        names = {c: (['FAS', 'TAS', 'SAS'][i] if selected == 3 else f"cluster_{i}") for i, c in enumerate(order)}
        result[domain] = {"selected_k": selected, "silhouette": scores,
                          "groups": {b.key: names[c] for b, c in zip(cells, labels)}}
    return {"purpose": "retrospective evaluation only; not valid for selection or training", "domains": result}


@torch.no_grad()
def benchmark(model, sequence_length, device, repetitions=100, warmup=10, use_gate=True):
    if repetitions < 1 or warmup < 0:
        raise ValueError("Invalid benchmark repetition count")
    x = torch.zeros(1, sequence_length, 4, device=device)
    model.eval()
    for _ in range(warmup):
        model.predict_soh(x, use_gate=use_gate)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        model.predict_soh(x, use_gate=use_gate)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {"device": str(device), "device_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "CPU",
            "mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "batch_size": 1,
            "sequence_length": sequence_length, "repetitions": repetitions,
            "scope": "inference path, domain classifier inactive; excludes preprocessing and I/O; local hardware only"}
