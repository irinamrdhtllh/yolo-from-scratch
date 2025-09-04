import torch
import torchvision
import torchvision.transforms as T


class LabelCodec:
    def __init__(self):
        self.label_map = {
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

    def encode(self, label: str) -> int:
        return self.label_map[label]

    def decode(self, encoded_label: int) -> str:
        for label in self.label_map:
            if self.label_map[label] == encoded_label:
                return label

        raise KeyError(f"Not a valid encoding: {encoded_label}")


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


def voc_to_yolo(target: dict):
    yolo_target = []

    annotation = target["annotation"]

    image_width = float(annotation["size"]["width"])
    image_height = float(annotation["size"]["height"])

    objects = annotation["object"]
    label_codec = LabelCodec()
    for obj in objects:
        name = label_codec.encode(obj["name"])

        xmin = float(obj["bndbox"]["xmin"])
        ymin = float(obj["bndbox"]["ymin"])
        xmax = float(obj["bndbox"]["xmax"])
        ymax = float(obj["bndbox"]["ymax"])

        x_center = (xmin + (xmax - xmin) / 2) / image_width
        y_center = (ymin + (ymax - ymin) / 2) / image_height
        box_width = (xmax - xmin) / image_width
        box_height = (ymax - ymin) / image_height

        yolo_target.append([name, x_center, y_center, box_width, box_height])

    return yolo_target


def encode_target(target, grid_size, bbox_per_cell, num_classes):
    out = torch.zeros(grid_size, grid_size, bbox_per_cell * 5 + num_classes)

    for obj in target:
        name, x_center, y_center, box_width, box_height = obj

        for i in range(grid_size):
            if i / 7 <= x_center < (i + 1) / 7:
                cell_x = i

        for i in range(grid_size):
            if i / 7 <= y_center < (i + 1) / 7:
                cell_y = i

        for i in range(bbox_per_cell):
            out[cell_y][cell_x][i * 5 + 0] = 1
            out[cell_y][cell_x][i * 5 + 1] = x_center
            out[cell_y][cell_x][i * 5 + 2] = y_center
            out[cell_y][cell_x][i * 5 + 3] = box_width
            out[cell_y][cell_x][i * 5 + 4] = box_height

        out[cell_y][cell_x][bbox_per_cell * 5 + name] = 1

    return out
