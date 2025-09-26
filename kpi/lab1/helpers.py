import numpy as np
import torch
import torch.nn.functional as F

def find_optimal_threshold(model, data_loader, device):
    model.eval()
    cos_similarities = []
    labels = []
    
    with torch.no_grad():
        for x1, x2, y in data_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            emb1, emb2 = model(x1, x2)
            cos_sim = F.cosine_similarity(emb1, emb2)
            cos_similarities.extend(cos_sim.cpu().numpy())
            labels.extend(y.cpu().numpy())

    best_threshold = 0
    best_accuracy = 0
    
    for threshold in np.arange(0, 1.0, 0.01):
        predictions = (np.array(cos_similarities) > threshold).astype(float)
        accuracy = np.mean(predictions == np.array(labels))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            
    print(f"Optimal threshold found: {best_threshold:.2f} with Test Accuracy: {best_accuracy:.4f}")
    return best_threshold
