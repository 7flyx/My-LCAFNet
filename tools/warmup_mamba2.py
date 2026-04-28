import argparse
import torch

from models.yolo_test import Model
from utils.general import check_file
from utils.torch_utils import select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="model.yaml path")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--img", type=int, default=320, help="image size")
    parser.add_argument("--iters", type=int, default=2, help="warmup iterations")
    opt = parser.parse_args()

    device = select_device(opt.device)
    cfg = check_file(opt.cfg)

    model = Model(cfg).to(device).eval()
    rgb = torch.randn(1, 3, opt.img, opt.img, device=device)
    ir = torch.randn(1, 3, opt.img, opt.img, device=device)

    with torch.no_grad():
        for _ in range(opt.iters):
            _ = model(rgb, ir)


if __name__ == "__main__":
    main()
