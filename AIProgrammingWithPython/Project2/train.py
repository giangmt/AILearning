# train.py
import argparse
import torch
from torchvision import models, datasets, transforms
from torch import nn, optim


def load_data(data_dir='flowers'):
    """
    データセットをロードしてデータ拡張と正規化を適用します。

    Args:
        data_dir (str): データセットのディレクトリ。

    Returns:
        tuple: トレーニング、検証、テストデータのデータローダーとデータセット。
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # トレーニングデータの変換（ランダムなリサイズ、クロップ、反転を含む）
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 検証およびテストデータの変換（リサイズおよびクロップ）
    validation_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # データセットの読み込み
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(
        valid_dir, transform=validation_test_transforms)
    test_data = datasets.ImageFolder(
        test_dir, transform=validation_test_transforms)

    # データローダーの作成
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return trainloader, validloader, testloader, train_data


def build_model(arch='vgg16', hidden_units=512, learning_rate=0.001):
    """
    事前訓練済みのネットワークをロードし、新しい分類器を構築してモデルを返します。

    Args:
        arch (str): 使用する事前訓練済みモデルのアーキテクチャ。
        hidden_units (int): 新しい分類器の隠れユニット数。
        learning_rate (float): オプティマイザの学習率。

    Returns:
        tuple: モデル、損失関数、オプティマイザ。
    """
    # 事前訓練済みモデルのロード
    model = getattr(models, arch)(pretrained=True)

    # パラメータの凍結
    for param in model.parameters():
        param.requires_grad = False

    # 新しい分類器の構築
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    # モデルに事前訓練済みモデルのアーキテクチャを設定
    model.arch = arch

    # モデルに新しい分類器を設定
    model.classifier = classifier

    # 損失関数とオプティマイザの定義
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer


def train_model(model, criterion, optimizer, dataloaders, epochs=5, device='cuda', save_dir='.'):
    """
    モデルをトレーニングします。

    Args:
        model (torch.nn.Module): トレーニングするモデル。
        criterion (torch.nn.Module): 損失関数。
        optimizer (torch.optim.Optimizer): オプティマイザ。
        dataloaders (tuple): トレーニング、検証、テストデータのデータローダーとデータセット。
        epochs (int): トレーニングエポック数。
        device (str): 使用するデバイス（'cuda'または'cpu'）。

    Returns:
        None
    """
    trainloader, validloader, _, train_data = dataloaders

    # モデルを指定されたデバイスに配置
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}")

        # Validation
        model.eval()
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()

                # Calculate accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}")

    # Save the final trained model
    checkpoint = {
        'arch': model.arch,
        'model_state_dict': model.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': train_data.class_to_idx
    }
    save_path = f'{save_dir}/checkpoint.pth'
    torch.save(checkpoint, save_path)
    print(f'Final trained model saved to {save_path}')


if __name__ == "__main__":
    # ArgumentParser オブジェクトを作成
    parser = argparse.ArgumentParser()

    # パラメーターを追加
    parser.add_argument('--data_dir', type=str,
                        default='flowers', help='データセットのディレクトリ')
    parser.add_argument('--save_dir', type=str,
                        default='.',  help='モデルのディレクトリ')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='使用する事前訓練済みモデルのアーキテクチャ')
    parser.add_argument('--hidden_units',
                        type=int, default=512, help='新しい分類器の隠れユニット数')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001, help='オプティマイザの学習率')
    parser.add_argument('--epochs',
                        type=int, default=5, help='トレーニングエポック数')
    parser.add_argument('--gpu', action='store_true', help='推論にGPUを使用')

    # コマンドラインを解析
    args = parser.parse_args()

    trainloader, validloader, _, train_data = load_data(args.data_dir)
    model, criterion, optimizer = build_model(
        args.arch, args.hidden_units, args.learning_rate)
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    train_model(model, criterion, optimizer, (trainloader,
                validloader, None, train_data), epochs=args.epochs, device=device, save_dir=args.save_dir)
