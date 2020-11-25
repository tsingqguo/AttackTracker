from torchvision import transforms
from extern.ssd_detect.utils import *
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import torch.nn.functional as F
import cv2

def detect(original_image, min_score, max_overlap, top_k, suppress=None,checkpoint=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
    if checkpoint is None:
        checkpoint = '/home/guoqing/projects/AttackTrack/extern/ssd_detect/model/checkpoint_ssd300.pth.tar'
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)

    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    # Transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #
    # import visdom
    # vis = visdom.Visdom()
    # vis.images(to_tensor(resize(original_image)))

    # Transform
    if isinstance(original_image,np.ndarray):
        original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
        original_image = Image.fromarray(original_image)
        image = normalize(to_tensor(resize(original_image)))
    else:
        tmp_img = F.interpolate(original_image, size=300).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        original_image = Image.fromarray(tmp_img.astype(np.uint8))
        image = normalize(to_tensor(tmp_img))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return [original_image.width//2-50,original_image.height//2-50,100,100]

    det_boxes[:, 0] = det_boxes[:, 0].clamp(0,original_image.width)
    det_boxes[:, 2] = det_boxes[:, 2].clamp(0, original_image.width)
    det_boxes[:, 1] = det_boxes[:, 1].clamp(0,original_image.height)
    det_boxes[:, 3] = det_boxes[:, 3].clamp(0, original_image.height)

    boxes = torch.zeros_like(det_boxes)
    boxes[:,0] = (det_boxes[:,0]+det_boxes[:,2])/2
    boxes[:,1] = (det_boxes[:,1]+det_boxes[:,3])/2
    boxes[:,2] = torch.abs(det_boxes[:,2]-det_boxes[:,0])
    boxes[:,3] = torch.abs(det_boxes[:,3]-det_boxes[:,1])

    return boxes.detach().numpy()

    # # Annotate
    # annotated_image = original_image
    # draw = ImageDraw.Draw(annotated_image)
    #
    # # Suppress specific classes, if needed
    # for i in range(det_boxes.size(0)):
    #     if suppress is not None:
    #         if det_labels[i] in suppress:
    #             continue
    #
    #     # Boxes
    #     box_location = det_boxes[i].tolist()
    #     draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
    #     draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
    #         det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
    #     # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
    #     #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
    #     # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
    #     #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness
    #
    #     # Text
    #     text_size = [15,15]
    #     text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
    #     textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
    #                         box_location[1]]
    #     draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
    #     draw.text(xy=text_location, text=det_labels[i].upper(), fill='white')
    # del draw
    #
    # return annotated_image
