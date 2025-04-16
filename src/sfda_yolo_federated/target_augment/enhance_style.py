def get_style_images(im_data, adain):
    """
    Apply style transfer to the input images

    Args:
        im_data: Input images (torch.Tensor), shape: [batch_size, channels, height, width]
        adain: The AdaIN style transfer model instance

    Returns:
        styled_im_data: Styled images (torch.Tensor)
    """
    # Apply style to the image using adain.add_style
    save_images = getattr(adain.args, "save_style_samples", False)
    styled_im_data = im_data * 0 + 1 * adain.add_style(
        im_data, 0, save_images=save_images
    )

    return styled_im_data
