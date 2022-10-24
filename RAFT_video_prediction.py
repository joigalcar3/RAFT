import sys
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from skimage.color import rgba2rgb

sys.path.append('core')
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder


DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    # if img.shape[2] == 4:
    #     # convert the image from RGBA2RGB
    #     img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)
    #     # img = rgba2rgb(img)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, video_file, step_viz):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    video_file.write(flo[:, :, [2, 1, 0]])

    if step_viz:
        img_flo = np.concatenate([img, flo], axis=0)

        # import matplotlib.pyplot as plt
        # plt.imshow(img_flo / 255.0)
        # plt.show()

        cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
        cv2.waitKey()
    return video_file


def RAFT_video_prediction(args, video_path, fps_out, img_start, img_step, step_viz=False):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        # images = [os.path.join(args.path, i) for i in os.listdir(args.path)]
        images = sorted(images)


        # Store video
        frameSize = tuple(load_image(images[0])[0].shape[2:0:-1])
        filename = os.path.join(video_path,
                                video_path.split("\\")[-1] + f"_RAFT_s{img_start}_f{fps_out}_k{img_step}.avi")
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), fps_out, frameSize)
        for imfile1, imfile2 in zip(images[img_start:-img_step], images[img_start+img_step:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            out = viz(image1, flow_up, out, step_viz)
        out.release()


if __name__ == '__main__':
    # User input
    demo_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_City_1024_576_2"
    # demo_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_city_256_144"
    # demo_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\RAFT\\demo-frames_2"
    demo_model = "models//raft-things.pth"
    video_storage_folder = os.path.join("D:\\AirSim simulator\\FDD\\Optical flow\\OpenCV_sparse\\video_storage",
                                        demo_folder.split("\\")[-1])
    fps_out = 30
    img_start = 30
    img_step = 1


    # Model input
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=demo_model, help="restore checkpoint")
    parser.add_argument('--path', default=demo_folder, help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
    args = parser.parse_args()

    RAFT_video_prediction(args, video_storage_folder, fps_out, img_start, img_step)
