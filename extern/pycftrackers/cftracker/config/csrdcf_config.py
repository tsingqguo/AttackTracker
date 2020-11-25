class CSRDCFConfig:
    # filter params
    padding=3
    interp_factor=0.02
    y_sigma=1
    channels_weight_lr=interp_factor
    use_channel_weights=True

    # segmentation params
    hist_lr=0.04
    nbins=16
    seg_colorspace='hsv' # 'bgr' or 'hsv'
    use_segmentation=True

    scale_type = 'normal'

    class ScaleConfig:
        scale_sigma_factor = 1 / 16.  # scale label function sigma
        scale_learning_rate = 0.025  # scale filter learning rate
        number_of_scales_filter = 33  # number of scales
        number_of_interp_scales = 33  # number of interpolated scales
        scale_model_factor = 1.0  # scaling of the scale model
        scale_step_filter = 1.02  # the scale factor of the scale sample patch
        scale_model_max_area = 32 * 16  # maximume area for the scale sample patch
        scale_feature = 'HOG4'  # features for the scale filter (only HOG4 supported)
        s_num_compressed_dim = 'MAX'  # number of compressed feature dimensions in the scale filter
        lamBda = 1e-2  # scale filter regularization
        do_poly_interp = False

    scale_config = ScaleConfig()

class CSRDCFLPConfig:
    # filter params
    padding=3
    interp_factor=0.02
    y_sigma=1
    channels_weight_lr=interp_factor
    use_channel_weights=True

    # segmentation params
    hist_lr=0.04
    nbins=16
    seg_colorspace='hsv' # 'bgr' or 'hsv'
    use_segmentation=True

    scale_type = 'LP'

    class ScaleConfig:
        learning_rate_scale = 0.015
        scale_sz_window = (128, 128)
        init_scale_factor=1.

    scale_config = ScaleConfig()



