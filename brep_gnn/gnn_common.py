"""Models, training loop and reporting shared by train_gnn.py (own brackets) and train_mfcad.py (MFCAD++)."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv


def one_hot(value, choices):
    return [1.0 if value == c else 0.0 for c in choices]


class FaceGNN(nn.Module):
    """Rounds of 'look at your neighbours and the links to them', then a classifier per face."""

    def __init__(self, n_in, n_edge, n_classes, hidden=64, layers=3):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.convs = nn.ModuleList(
            [GATv2Conv(hidden, hidden // 4, heads=4, edge_dim=n_edge) for _ in range(layers)])
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, data):
        h = F.relu(self.inp(data.x))
        for conv in self.convs:
            h = h + F.relu(conv(h, data.edge_index, data.edge_attr))
        return self.out(h)


class FaceMLP(nn.Module):
    """Same size, but each face is judged alone: no neighbours, no links."""

    def __init__(self, n_in, n_classes, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, data):
        return self.net(data.x)


def evaluate(model, loader, device, n_classes):
    """Return the confusion matrix: rows = true label, columns = predicted label."""
    model.eval()
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=1)
            idx = batch.y * n_classes + pred
            cm += torch.bincount(idx, minlength=n_classes * n_classes).view(n_classes, n_classes).cpu()
    return cm


def accuracy(cm):
    return cm.diag().sum().item() / cm.sum().item()


def mean_iou(cm):
    """Intersection-over-union per class, averaged over the classes that occur."""
    tp = cm.diag().float()
    union = cm.sum(0).float() + cm.sum(1).float() - tp
    present = cm.sum(1) > 0
    return (tp[present] / union[present]).mean().item()


def train(model, train_loader, val_loader, epochs, device, name, n_classes, weights=None, lr=3e-3, print_every=25):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y, weight=weights)
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
        val_acc = accuracy(evaluate(model, val_loader, device, n_classes))
        if val_acc > best_acc:
            best_acc, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % print_every == 0 or epoch == 1:
            print(f"  [{name}] epoch {epoch:>3}  loss {total / len(train_loader.dataset):.4f}  val acc {val_acc:.3f}")
    model.load_state_dict(best_state)
    return model


def report(name, cm, labels):
    print(f"\n{name}: test accuracy {100 * accuracy(cm):.2f}%, mean IoU {100 * mean_iou(cm):.2f}% over {cm.sum().item()} faces")
    if len(labels) <= 5:
        print(f"  {'true \\ predicted':<18}" + "".join(f"{l:>8}" for l in labels) + "   per-class accuracy")
        for i, label in enumerate(labels):
            row = cm[i]
            recall = 100 * row[i].item() / max(1, row.sum().item())
            print(f"  {label:<18}" + "".join(f"{v:>8}" for v in row.tolist()) + f"   {recall:6.2f}%")
    else:
        print(f"  {'class':<38}{'faces':>8}  accuracy      IoU")
        for i, label in enumerate(labels):
            n = cm[i].sum().item()
            if n == 0:
                continue
            tp = cm[i, i].item()
            union = n + cm[:, i].sum().item() - tp
            print(f"  {label:<38}{n:>8}  {100 * tp / n:7.2f}%  {100 * tp / union:6.2f}%")
