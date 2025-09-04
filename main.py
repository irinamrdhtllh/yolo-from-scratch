import dataset
import models


def calculate_loss(): ...


def main():
    data = dataset.load_dataset()
    image, target = data[0]
    print(target)
    print(image.shape)

    target = dataset.voc_to_yolo(target)
    print(target)

    encoded_target = dataset.encode_target(
        target, grid_size=7, bbox_per_cell=2, num_classes=20
    )
    print(encoded_target[:, :, 0])

    model = models.YOLOv1(grid_size=7, bbox_per_cell=2, num_classes=20)
    out = model(image)
    print(out.shape)


if __name__ == "__main__":
    main()
