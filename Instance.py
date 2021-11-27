import pixellib
from pixellib.instance import instance_segmentation

segment_image = instance_segmentation()

segment_image.load_model("models/mask_r_cnninstance.h5")

segment_image.segmentImage('input/98_tributo.jpg', output_image_name='output/instance/98_tributo.jpg')
