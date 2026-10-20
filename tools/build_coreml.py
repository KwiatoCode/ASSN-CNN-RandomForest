import argparse
import copy
import json
import random
import string
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / 'ASSN' / 'ASLClassifier.json'
DEFAULT_BACKBONE_OUTPUT = ROOT / 'ASSN' / 'ASLBackbone.mlpackage'
DEFAULT_ARTIFACTS = ROOT / 'tools' / 'trained'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class ASLBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, image):
        return torch.flatten(self.features(image), 1)


class ASLClassifier(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.backbone = ASLBackbone()
        self.head = nn.Linear(256, num_classes)

    def forward(self, image):
        return self.head(self.backbone(image))


class ASLDataset(Dataset):
    def __init__(self, paths, labels, augmentations=0, image_size=224):
        self.paths = list(paths)
        self.labels = list(labels)
        self.augmentations = augmentations
        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.augment = transforms.RandomAffine(
            degrees=12,
            translate=(0.08, 0.08),
            scale=(0.85, 1.15),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=(255, 255, 255),
        )

    def __len__(self):
        return len(self.paths) * (self.augmentations + 1)

    def __getitem__(self, index):
        source_index = index % len(self.paths)
        with Image.open(self.paths[source_index]) as source:
            image = source.convert('RGB')
        if index >= len(self.paths):
            image = self.augment(image)
        return self.preprocess(image), self.labels[source_index]


def collect_samples(dataset, count):
    root = Path(dataset).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Nie znaleziono datasetu: {root}')
    candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir())
    selected = None
    for parent in candidates:
        class_dirs = [p for p in parent.iterdir() if p.is_dir() and p.name.upper() in string.ascii_uppercase and len(p.name) == 1]
        if len(class_dirs) >= 2:
            selected = class_dirs
            break
    if selected is None:
        raise ValueError('Dataset musi mieć podfoldery A, B, ..., Z z obrazami .png/.jpg/.jpeg/.bmp/.webp')
    paths, labels = [], []
    for folder in sorted(selected, key=lambda p: p.name.upper()):
        files = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if count:
            files = files[:count]
        if len(files) < 5:
            raise ValueError(f'Klasa {folder.name} ma mniej niż 5 obrazów.')
        paths.extend(str(p) for p in files)
        labels.extend([folder.name.upper()] * len(files))
        print(f'{folder.name.upper()}: {len(files)} obrazów')
    if len(set(labels)) != 26:
        raise ValueError(f'Oczekiwano 26 klas A–Z, znaleziono {len(set(labels))}.')
    return paths, labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loader(paths, labels, batch_size, augmentations=0, shuffle=False, workers=0, device=None, image_size=224):
    return DataLoader(
        ASLDataset(paths, labels, augmentations=augmentations, image_size=image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device is not None and device.type == 'cuda',
    )


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, total_correct / total


def train_cnn(train_paths, train_labels, val_paths, val_labels, class_names, args, device, artifacts):
    lookup = {label: index for index, label in enumerate(class_names)}
    train_idx = [lookup[y] for y in train_labels]
    val_idx = [lookup[y] for y in val_labels]
    train_loader = create_loader(train_paths, train_idx, args.batch_size, args.augmentations, True, args.workers, device, args.image_size)
    val_loader = create_loader(val_paths, val_idx, args.batch_size, workers=args.workers, device=device, image_size=args.image_size)
    model = ASLClassifier(len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    best_state = None
    stale = 0
    metrics = []
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        val_loss, val_accuracy = evaluate(model, val_loader, device, criterion)
        entry = {
            'epoch': epoch + 1,
            'train_loss': total_loss / total,
            'train_accuracy': total_correct / total,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }
        metrics.append(entry)
        print(f'Epoka {entry["epoch"]}/{args.epochs}: train_loss={entry["train_loss"]:.4f} train_accuracy={entry["train_accuracy"]:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}', flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    artifacts.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.backbone.state_dict(), 'classes': class_names, 'architecture': 'ASLBackbone', 'image_size': args.image_size}, artifacts / 'cnn.pt')
    with open(artifacts / 'history.json', 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)
    return model


def embeddings(backbone, loader, device):
    backbone.eval()
    features, labels = [], []
    with torch.inference_mode():
        for images, batch_labels in loader:
            features.append(backbone(images.to(device)).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def export_randomforest(randomforest, output):
    trees = []
    for estimator in randomforest.estimators_:
        tree = estimator.tree_
        leaf = [-1] * tree.node_count
        probabilities = []
        for index in range(tree.node_count):
            if tree.children_left[index] == -1:
                values = np.asarray(tree.value[index, 0], dtype=np.float64)
                leaf[index] = len(probabilities)
                probabilities.append((values / values.sum()).tolist())
        trees.append({
            'feature': tree.feature.tolist(),
            'threshold': tree.threshold.tolist(),
            'left': tree.children_left.tolist(),
            'right': tree.children_right.tolist(),
            'leaf': leaf,
            'probabilities': probabilities,
        })
    data = {
        'classes': [str(label) for label in randomforest.classes_],
        'featureCount': int(randomforest.n_features_in_),
        'trees': trees,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as file:
        json.dump(data, file, separators=(',', ':'), allow_nan=False)
    print(f'Zapisano {output} ({len(trees)} drzew, {data["featureCount"]} cech)')


def export_backbone(checkpoint, output):
    import coremltools as ct

    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = ASLBackbone()
    model.load_state_dict(saved['state_dict'])
    model.eval()
    image_size = int(saved.get('image_size', 224))
    traced = torch.jit.trace(model, torch.rand(1, 3, image_size, image_size))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name='image', shape=(1, 3, image_size, image_size), scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name='embedding')],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.short_description = 'ASL A–Z: CNN trenowana od podstaw na zdjęciach szkieletów dłoni'
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))
    print(f'Zapisano {output}')


def train(args):
    set_seed(args.seed)
    paths, labels = collect_samples(args.train, args.count)
    class_names = sorted(set(labels))
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=0.25, random_state=args.seed, stratify=labels
    )
    validation_fraction = max(0.1, len(class_names) / len(train_val_paths))
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=validation_fraction, random_state=args.seed, stratify=train_val_labels
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Urządzenie: {device} | train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}')
    artifacts = Path(args.artifacts).expanduser().resolve()
    model = train_cnn(train_paths, train_labels, val_paths, val_labels, class_names, args, device, artifacts)
    lookup = {label: index for index, label in enumerate(class_names)}
    test_loader = create_loader(test_paths, [lookup[y] for y in test_labels], args.batch_size, workers=args.workers, device=device, image_size=args.image_size)
    test_loss, test_accuracy = evaluate(model, test_loader, device, nn.CrossEntropyLoss())
    train_loader = create_loader(train_paths, [lookup[y] for y in train_labels], args.batch_size, args.augmentations, workers=args.workers, device=device, image_size=args.image_size)
    x_train, y_train = embeddings(model.backbone, train_loader, device)
    x_test, y_test = embeddings(model.backbone, test_loader, device)
    forest_labels = np.asarray([class_names[int(index)] for index in y_train])
    randomforest = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=24,
        min_samples_leaf=2,
        random_state=args.seed,
        n_jobs=-1,
    ).fit(x_train, forest_labels)
    predictions = randomforest.predict(x_test)
    test_class_names = np.asarray([class_names[int(index)] for index in y_test])
    results = {
        'classes': class_names,
        'train_samples': len(train_paths),
        'validation_samples': len(val_paths),
        'test_samples': len(test_paths),
        'augmentations': args.augmentations,
        'image_size': args.image_size,
        'dataset_samples': len(paths),
        'seed': args.seed,
        'cnn_accuracy_test': test_accuracy,
        'cnn_loss_test': test_loss,
        'randomforest_accuracy_test': accuracy_score(test_class_names, predictions),
        'randomforest_f1_weighted_test': f1_score(test_class_names, predictions, average='weighted'),
        'randomforest_classification_report': classification_report(test_class_names, predictions, labels=class_names, output_dict=True, zero_division=0),
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(randomforest, artifacts / 'randomforest.pkl')
    with (artifacts / 'results.json').open('w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)
    print(f'CNN accuracy (test): {test_accuracy:.4f}')
    print(f'Random Forest accuracy (test): {results["randomforest_accuracy_test"]:.4f}')
    print(f'Random Forest F1 (test): {results["randomforest_f1_weighted_test"]:.4f}')
    export_randomforest(randomforest, args.output)
    if not args.skip_coreml:
        export_backbone(artifacts / 'cnn.pt', args.backbone_output)
    return results


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--train', help='folder datasetu z podfolderami A–Z')
    source.add_argument('--checkpoint', help='zapisany checkpoint CNN: cnn.pt')
    parser.add_argument('--randomforest', help='plik randomforest.pkl wymagany z --checkpoint')
    parser.add_argument('--count', type=int, default=0)
    parser.add_argument('--augmentations', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--trees', type=int, default=160)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--artifacts', default=str(DEFAULT_ARTIFACTS))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--backbone-output', default=str(DEFAULT_BACKBONE_OUTPUT))
    parser.add_argument('--skip-coreml', action='store_true')
    args = parser.parse_args()
    if (args.count != 0 and args.count < 5) or args.image_size < 32 or args.augmentations < 0 or args.epochs < 1 or args.batch_size < 1 or args.trees < 1 or args.patience < 1 or args.workers < 0:
        parser.error('Nieprawidłowe parametry treningu.')
    if args.train:
        if args.randomforest:
            parser.error('--randomforest można używać tylko razem z --checkpoint.')
        train(args)
    else:
        if not args.randomforest:
            parser.error('--checkpoint wymaga --randomforest.')
        randomforest = joblib.load(args.randomforest)
        export_randomforest(randomforest, args.output)
        if not args.skip_coreml:
            export_backbone(args.checkpoint, args.backbone_output)


if __name__ == '__main__':
    main()
