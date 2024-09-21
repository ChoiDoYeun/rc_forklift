import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import time

# 모델 정의 함수 (ResNet, MobileNet 등을 미리 정의)
from torchvision.models import resnet18, resnet50, mobilenet_v2, mobilenet_v3_small

# 모델 선택 함수
def create_model(model_name):
    if model_name == 'resnet18':
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)  # 클래스 수에 맞게 출력 레이어 조정
    elif model_name == 'resnet50':
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'mobilenet_v2':
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model_name == 'mobilenet_v3_small':
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

# 모델 불러오기 및 검증 함수
def evaluate_model(model_name, model_path):
    # 모델 불러오기
    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    val_preds = []
    val_labels = []
    
    # FPS 측정을 위한 시간 계산
    start_time = time.time()
    total_images = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            # 모델 추론
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            total_images += len(images)  # 처리한 이미지 개수 계산

    # F1 score 계산
    val_f1 = f1_score(val_labels, val_preds, average='binary')
    print(f"F1 Score for {model_name}: {val_f1:.4f}")

    # FPS 계산
    end_time = time.time()
    total_time = end_time - start_time
    fps = total_images / total_time
    print(f"FPS for {model_name}: {fps:.2f} frames per second")
    
    return val_f1, fps

# 프루닝 전후 모델을 불러와 성능을 비교하는 함수
def compare_pruned_and_unpruned_models(model_name, unpruned_model_path, pruned_model_path):
    print(f"\nEvaluating unpruned {model_name} model...")
    unpruned_f1, unpruned_fps = evaluate_model(model_name, unpruned_model_path)

    print(f"\nEvaluating pruned {model_name} model...")
    pruned_f1, pruned_fps = evaluate_model(model_name, pruned_model_path)

    # 결과 비교
    print(f"\n--- Comparison for {model_name} ---")
    print(f"Unpruned model F1 Score: {unpruned_f1:.4f}, FPS: {unpruned_fps:.2f}")
    print(f"Pruned model F1 Score: {pruned_f1:.4f}, FPS: {pruned_fps:.2f}")
    print(f"FPS improvement: {pruned_fps - unpruned_fps:.2f} frames per second")

# 각 모델의 프루닝 전후 경로 설정
unpruned_model_paths = {
    'resnet18': 'best_resnet18_model.pth',
    'resnet50': 'best_resnet50_model.pth',
    'mobilenet_v2': 'best_mobilenet_v2_model.pth',
    'mobilenet_v3_small': 'best_mobilenet_v3_small_model.pth'
}

pruned_model_paths = {
    'resnet18': 'pruned_resnet18_model.pth',
    'resnet50': 'pruned_resnet50_model.pth',
    'mobilenet_v2': 'pruned_mobilenet_v2_model.pth',
    'mobilenet_v3_small': 'pruned_mobilenet_v3_small_model.pth'
}

# 모델 리스트에서 각각 프루닝 전후 성능을 비교
for model_name in unpruned_model_paths.keys():
    compare_pruned_and_unpruned_models(model_name, unpruned_model_paths[model_name], pruned_model_paths[model_name])
