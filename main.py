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
    bbox_coord_loss = 0
    bbox_size_loss = 0
    confidence_loss = 0
    confidence_noobj_loss = 0
    class_loss = 0

    for i in range(grid_size):
        for j in range(grid_size):
            for b in range(bbox_per_cell):
                target_obj_prob = target[i, j, b * 5]
                target_x_center = target[i, j, b * 5 + 1]
                target_y_center = target[i, j, b * 5 + 2]
                target_width = target[i, j, b * 5 + 3]
                target_height = target[i, j, b * 5 + 4]

                pred_obj_prob = pred[i, j, b * 5]
                pred_x_center = pred[i, j, b * 5 + 1]
                pred_y_center = pred[i, j, b * 5 + 2]
                pred_width = pred[i, j, b * 5 + 3]
                pred_height = pred[i, j, b * 5 + 4]

                if target_obj_prob > 0:
                    # fmt: off
                    bbox_coord_loss += ((target_x_center - pred_x_center) ** 2 
                                        + (target_y_center - pred_y_center) ** 2)
                    bbox_size_loss += ((torch.sqrt(torch.abs(target_width)) - torch.sqrt(torch.abs(pred_width))) ** 2 
                                       + (torch.sqrt(torch.abs(target_height)) - torch.sqrt(torch.abs(pred_height))) ** 2)
                    confidence_loss += (target_obj_prob - pred_obj_prob) ** 2
                    # fmt: on

                    for c in range(num_classes):
                        target_class = target[i][j][bbox_per_cell * 5 + c]
                        pred_class = pred[i][j][bbox_per_cell * 5 + c]
                        class_loss += (target_class - pred_class) ** 2
                else:
                    confidence_noobj_loss += (target_obj_prob - pred_obj_prob) ** 2

    total_loss = (
        lambda_coord * bbox_coord_loss
        + lambda_coord * bbox_size_loss
        + confidence_loss
        + lambda_noobj * confidence_noobj_loss
        + class_loss
    )

    return total_loss


def train(model, data, num_epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )

    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(len(data)):
            image, target = data[i]

            target = dataset.voc_to_yolo(target)
            encoded_target = dataset.encode_target(
                target, grid_size=7, bbox_per_cell=2, num_classes=20
            )

            image = image.to(device)
            encoded_target = encoded_target.to(device)

            pred = model(image)

            loss = calculate_loss(
                pred,
                encoded_target,
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

        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {total_loss/len(data):.4f}")


def main():
    model = models.YOLOv1(grid_size=7, bbox_per_cell=2, num_classes=20)
    data = dataset.load_dataset()
    train(model, data, num_epochs=10, lr=0.001, device="cuda")


if __name__ == "__main__":
    main()
