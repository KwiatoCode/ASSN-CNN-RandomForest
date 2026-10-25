from pathlib import Path
import argparse
import torch
from torch import nn


class ASLBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, image):
        return torch.flatten(self.features(image), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='tools/trained/cnn.pt')
    parser.add_argument('--output', default='web/models/backbone.onnx')
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = ASLBackbone().eval()
    model.load_state_dict(saved['state_dict'])
    image_size = int(saved.get('image_size', 224))
    if image_size != 224:
        raise ValueError(f'Model ma wejście {image_size} x {image_size}, a witryna wymaga 224 x 224.')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(model, sample, str(target), export_params=True, opset_version=17,
                      input_names=['image'], output_names=['embedding'], do_constant_folding=True, dynamo=False)
    import onnx
    import numpy as np
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(target)))
    expected = model(sample).detach().numpy()
    session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
    actual = session.run(['embedding'], {'image': sample.numpy()})[0]
    if expected.shape != (1, 256) or actual.shape != (1, 256):
        raise RuntimeError(f'Niepoprawny embedding: PyTorch {expected.shape}, ONNX {actual.shape}')
    diff = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1e-3, atol=1e-4):
        raise RuntimeError(f'Niezgodne wyniki PyTorch i ONNX, największa różnica: {diff}')
    print(f'ONNX OK: {target} ({target.stat().st_size} bajtów); maksymalna różnica: {diff:.8f}')


if __name__ == '__main__':
    main()
