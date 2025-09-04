import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as T


class YOLOv1(nn.Module):
    """The first YOLO architecture proposed in the paper titled
    "You Only Look Once: Unified, Real-Time Object Detection" by Redmon et al."""

    def __init__(self, grid_size, bbox_per_cell, num_classes):
        super().__init__()

        self.grid_size = grid_size
        self.bbox_per_cell = bbox_per_cell
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=192, kernel_size=7, stride=2, padding=3
            ),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.layer7 = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 1024, out_features=4096),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.layer8 = nn.Linear(
            in_features=4096,
            out_features=grid_size * grid_size * (bbox_per_cell * 5 + num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(-1)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.view(
            self.grid_size,
            self.grid_size,
            (self.bbox_per_cell * 5 + self.num_classes),
        )

        return x


def load_dataset():
    transform = T.Compose(
        [
            T.Resize((448, 448)),
            T.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.VOCDetection(
        root="data",
        year="2007",
        image_set="trainval",
        transform=transform,
    )

    return dataset


def class_to_label(class_name: str):
    class_map = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "diningtable": 10,
        "dog": 11,
        "horse": 12,
        "motorbike": 13,
        "person": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19,
    }

    return class_map.get(class_name, 20)


def label_to_class(label: int):
    label_map = {
        0: "aeroplane",
        1: "bicycle",
        2: "bird",
        3: "boat",
        4: "bottle",
        5: "bus",
        6: "car",
        7: "cat",
        8: "chair",
        9: "cow",
        10: "diningtable",
        11: "dog",
        12: "horse",
        13: "motorbike",
        14: "person",
        15: "pottedplant",
        16: "sheep",
        17: "sofa",
        18: "train",
        19: "tvmonitor",
    }

    return label_map.get(label)


def voc_to_yolo(target: dict):
    labels = []

    annotation = target["annotation"]

    image_width = float(annotation["size"]["width"])
    image_height = float(annotation["size"]["height"])

    objects = annotation["object"]
    for o in objects:
        name = o["name"]
        label = class_to_label(name)

        xmin = float(o["bndbox"]["xmin"])
        ymin = float(o["bndbox"]["ymin"])
        xmax = float(o["bndbox"]["xmax"])
        ymax = float(o["bndbox"]["ymax"])

        x_center = (xmin + (xmax - xmin) / 2) / image_width
        y_center = (ymin + (ymax - ymin) / 2) / image_height
        box_width = (xmax - xmin) / image_width
        box_height = (ymax - ymin) / image_height

        labels.append([label, x_center, y_center, box_width, box_height])

    return labels


def main():
    dataset = load_dataset()
    image, target = dataset[0]
    print(image.shape)

    labels = voc_to_yolo(target)
    print(labels)

    model = YOLOv1(grid_size=7, bbox_per_cell=2, num_classes=20)
    out = model(image)
    print(out.shape)


if __name__ == "__main__":
    main()
