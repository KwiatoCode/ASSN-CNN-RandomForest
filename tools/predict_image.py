import argparse
from pathlib import Path

import joblib
import torch
from PIL import Image
from torchvision import transforms

from build_coreml import ASLBackbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', default='tools/trained/cnn.pt')
    parser.add_argument('--randomforest', default='tools/trained/randomforest.pkl')
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = ASLBackbone()
    model.load_state_dict(saved['state_dict'])
    model.eval()
    image_size = int(saved.get('image_size', 224))
    preprocessing = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    with Image.open(args.image) as source:
        image = preprocessing(source.convert('RGB')).unsqueeze(0)

    with torch.inference_mode():
        embedding = model(image).numpy()
    forest = joblib.load(args.randomforest)
    probabilities = forest.predict_proba(embedding)[0]
    best = probabilities.argsort()[::-1][:3]
    for rank, index in enumerate(best, start=1):
        print(f'{rank}. {forest.classes_[index]}: {probabilities[index] * 100:.2f}%')


if __name__ == '__main__':
    main()
