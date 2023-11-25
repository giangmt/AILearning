# predict.py
from PIL import Image
import argparse
import numpy as np
import json
import torch
from torchvision import models, datasets, transforms
from torch import nn, optim


def load_checkpoint(filepath):
    """
    保存されたモデルのチェックポイントをロードしてモデルを構築します。

    Args:
        filepath (str): チェックポイントファイルのパス。

    Returns:
        torch.nn.Module: ロードされたモデル
    """
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def process_image(image):
    """
    PIL 画像を PyTorch モデル用にスケーリング、クロップ、および正規化します。

    Args:
        image (PIL.Image): モデルに使用する PIL 画像。

    Returns:
        numpy.ndarray: モデル用に前処理された Numpy 配列。
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(image)
    return img_tensor


def predict(image_path, model, category_names, topk=5, device='cuda'):
    """
    トレーニングされたネットワークを使用して画像ファイルからクラスを予測します。

    Args:
        image_path (str): 予測する画像のファイルパス。
        model (torch.nn.Module): トレーニング済みモデル。
        topk (int): 返されるトップ K の確率とクラス数。

    Returns:
        tuple: 予測された確率のリストとクラスのリスト。
    """
    img = Image.open(image_path)
    processed_img = process_image(img)
    img_tensor = processed_img.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    # モデルを指定されたデバイスに配置
    model.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(img_tensor)

    probs, indices = torch.topk(
        torch.nn.functional.softmax(outputs, dim=1), topk)

    probs = probs.cpu().numpy().flatten()
    indices = indices.cpu().numpy().flatten()

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]

    if category_names is None:
        return probs, classes
    else:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)

        class_names = [cat_to_name[class_label] for class_label in classes]
        return probs, class_names


if __name__ == "__main__":
    # ArgumentParser オブジェクトを作成
    parser = argparse.ArgumentParser()

    # パラメーターを追加
    parser.add_argument('--image_path', type=str,
                        default='./flowers/test/1/image_06743.jpg', help='予測する画像のファイルパス')
    parser.add_argument('--topk', type=int,
                        default=5, help='返されるトップKの確率とクラス数')
    parser.add_argument('--category_names', type=str, help='カテゴリ名のファイル')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth', help='モデルを定指')
    parser.add_argument('--gpu', action='store_true', help='推論にGPUを使用')

    # コマンドラインを解析
    args = parser.parse_args()

    loaded_model = load_checkpoint(args.checkpoint)
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    probs, classes = predict(
        args.image_path, loaded_model, args.category_names, topk=args.topk, device=device)

    # Display top predictions
    for prob, class_ in zip(probs, classes):
        print(f"Class: {class_}, Probability: {prob}")
