import os

import torch
from torch.utils.data import DataLoader

import dataset
import models


def calculate_loss(
    pred,
    target,
    grid_size: int,
    bbox_per_cell: int,
    num_classes: int,
    lambda_coord: int,
    lambda_noobj: int,
):
    batch_size = pred.shape[0]

    target_bbox = target[:, :, :, 0 : bbox_per_cell * 5]
    target_bbox = target_bbox.reshape(
        batch_size, grid_size, grid_size, bbox_per_cell, 5
    )

    pred_bbox = pred[:, :, :, 0 : bbox_per_cell * 5]
    pred_bbox = pred_bbox.reshape(batch_size, grid_size, grid_size, bbox_per_cell, 5)

    target_prob = target_bbox[:, :, :, :, 0]
    target_coord = target_bbox[:, :, :, :, 1:3]
    target_size = target_bbox[:, :, :, :, 3:5]

    pred_prob = pred_bbox[:, :, :, :, 0]
    pred_coord = pred_bbox[:, :, :, :, 1:3]
    pred_size = pred_bbox[:, :, :, :, 3:5]

    bbox_coord_loss = lambda_coord * torch.sum(
        target_prob.unsqueeze(-1) * (target_coord - pred_coord) ** 2
    )
    bbox_size_loss = lambda_coord * torch.sum(
        target_prob.unsqueeze(-1)
        * (torch.sqrt(target_size) - torch.sqrt(torch.clamp(pred_size, min=1e-6))) ** 2
    )
    confidence_loss = torch.sum(target_prob * (target_prob - pred_prob) ** 2)
    confidence_noobj_loss = lambda_noobj * torch.sum(
        (1 - target_prob) * (target_prob - pred_prob) ** 2
    )

    target_prob = (
        target_prob.unsqueeze(-1)
        .expand(-1, -1, -1, -1, 10)
        .reshape(batch_size, grid_size, grid_size, num_classes)
    )
    target_class = target[:, :, :, bbox_per_cell * 5 :]
    pred_class = pred[:, :, :, bbox_per_cell * 5 :]
    class_loss = torch.sum(target_prob * (target_class - pred_class) ** 2)

    total_loss = (
        bbox_coord_loss
        + bbox_size_loss
        + confidence_loss
        + confidence_noobj_loss
        + class_loss
    ) / batch_size

    return total_loss


def train(model, data, num_epochs, batch_size, lr, device):
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)

            loss = calculate_loss(
                preds,
                targets,
                grid_size=7,
                bbox_per_cell=2,
                num_classes=20,
                lambda_coord=5,
                lambda_noobj=0.5,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss,
            },
            f"checkpoints/yolov1_epoch_{epoch + 1}.pth",
        )

        print(
            f"Epoch: {epoch + 1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}"
        )


def validate(model, data, device): ...


def main():
    torch.autograd.set_detect_anomaly(True)
    os.makedirs("checkpoints", exist_ok=True)

    model = models.YOLOv1(grid_size=7, bbox_per_cell=2, num_classes=20)

    print("Loading the dataset.")
    data = dataset.load_dataset()

    print("Starting training.")
    train(model, data, num_epochs=10, batch_size=2, lr=0.001, device="cuda")


if __name__ == "__main__":
    main()
