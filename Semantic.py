import pixellib
from pixellib.semantic import semantic_segmentation

segment_image = semantic_segmentation()

segment_image.load_pascalvoc_model('models/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')

segment_image.segmentAsPascalvoc('input/98_tributo.jpg', output_image_name='output/98_tributo.jpg')
