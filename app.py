
# Import libraries
import pixellib
from pixellib.instance import instance_segmentation

# Download model
# https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5

segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5")
segment_image.segmentImage("img.jpeg", show_bboxes=True, output_image_name='output1.jpeg')
