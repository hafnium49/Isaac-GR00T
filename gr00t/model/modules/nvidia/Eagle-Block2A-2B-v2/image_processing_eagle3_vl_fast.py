# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/image_processing_llava_onevision_fast.py

# Check if fast image processing is available (requires newer transformers)
_FAST_IMAGE_PROCESSING_AVAILABLE = False
try:
    from transformers.image_processing_utils_fast import (
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        BaseImageProcessorFast,
        DefaultFastImageProcessorKwargs,
        divide_to_patches,
        group_images_by_shape,
        reorder_images,
    )
    _FAST_IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    # Older transformers version - fast processing not available
    # Define stub classes so the module can be imported without errors
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = ""
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = ""

    # Create a fallback base class with necessary methods
    from transformers.image_processing_utils import BaseImageProcessor
    class BaseImageProcessorFast(BaseImageProcessor):
        """Fallback class when fast image processing is not available."""
        @classmethod
        def register_for_auto_class(cls, auto_class="AutoImageProcessor"):
            pass  # No-op for compatibility

    DefaultFastImageProcessorKwargs = dict
    divide_to_patches = None
    group_images_by_shape = None
    reorder_images = None

from typing import List, Optional, Union

from transformers.image_processing_utils import BatchFeature, get_patch_output_size, select_best_resolution
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    IMAGENET_STANDARD_MEAN, # 0.5, 0.5, 0.5
    IMAGENET_STANDARD_STD, # 0.5, 0.5, 0.5
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
)
# These may not exist in older transformers versions
try:
    from transformers.image_utils import VideoInput
except ImportError:
    VideoInput = List  # Fallback type
try:
    from transformers.image_utils import SizeDict
except ImportError:
    SizeDict = dict  # Fallback type
try:
    from transformers.image_utils import make_flat_list_of_images
except ImportError:
    def make_flat_list_of_images(images):
        if isinstance(images, list):
            return images
        return [images]
try:
    from transformers.image_utils import make_batched_videos
except ImportError:
    make_batched_videos = None
try:
    from transformers.image_utils import validate_kwargs
except ImportError:
    def validate_kwargs(captured_kwargs, valid_processor_keys):
        pass  # No-op fallback
from transformers.processing_utils import Unpack
from transformers.utils import TensorType, add_start_docstrings, is_torch_available, is_torchvision_v2_available


if is_torch_available():
    import torch
if is_torchvision_v2_available():
    try:
        from transformers.image_utils import pil_torch_interpolation_mapping
    except ImportError:
        # Fallback mapping for older transformers
        from torchvision.transforms import InterpolationMode
        pil_torch_interpolation_mapping = {
            PILImageResampling.NEAREST: InterpolationMode.NEAREST,
            PILImageResampling.BILINEAR: InterpolationMode.BILINEAR,
            PILImageResampling.BICUBIC: InterpolationMode.BICUBIC,
            PILImageResampling.LANCZOS: InterpolationMode.LANCZOS,
            PILImageResampling.BOX: InterpolationMode.BOX,
        }
    from torchvision.transforms.v2 import functional as F
else:
    from torchvision.transforms import functional as F
    # Define fallback mapping
    from torchvision.transforms import InterpolationMode
    pil_torch_interpolation_mapping = {
        PILImageResampling.NEAREST: InterpolationMode.NEAREST,
        PILImageResampling.BILINEAR: InterpolationMode.BILINEAR,
        PILImageResampling.BICUBIC: InterpolationMode.BICUBIC,
        PILImageResampling.LANCZOS: InterpolationMode.LANCZOS,
        PILImageResampling.BOX: InterpolationMode.BOX,
    }

def crop(img: torch.Tensor, left: int, top: int, right: int, bottom: int) -> torch.Tensor:
    """Crop the given numpy array.
    
    Args:
        img (torch.Tensor): Image to be cropped. Format should be (C, H, W).
        left (int): The left coordinate of the crop box.
        top (int): The top coordinate of the crop box.
        right (int): The right coordinate of the crop box.
        bottom (int): The bottom coordinate of the crop box.
        
    Returns:
        torch.Tensor: Cropped image.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError('img should be torch.Tensor. Got {}'.format(type(img)))
    
    if img.ndim not in [2, 3]:
        raise ValueError('Image should have 2 or 3 dimensions. Got {}'.format(img.ndim))
    
    img_height = img.shape[1]
    img_width = img.shape[2]
    if top < 0 or left < 0 or bottom > img_height or right > img_width:
        raise ValueError('Crop coordinates out of bounds')
    
    if top >= bottom or left >= right:
        raise ValueError('Invalid crop coordinates')

    return img[:, top:bottom, left:right]


class Eagle3_VLFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_pad: Optional[bool]


@add_start_docstrings(
    "Constructs a fast ConvNeXT image processor. Based on [`SiglipImageProcessor`] with incorporation of processing each video frame.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        image_grid_pinpoints (`List[List[int]]`, *optional*):
            A list of possible resolutions to use for processing high resolution images. The best resolution is selected
            based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
            method. Not used for processing videos.
        do_pad (`bool`, *optional*):
            Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
            number of patches in the batch. Padding will be applied to the bottom and right with zeros.
    """,
)
class Eagle3_VLImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 448, "width": 448}
    default_to_square = False
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    valid_kwargs = Eagle3_VLFastImageProcessorKwargs
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs: Unpack[Eagle3_VLFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
            do_pad (`bool`, *optional*):
                    Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                    number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        """,
    )
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Eagle3_VLFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _prepare_images_structure(
        self,
        images: ImageInput,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        return make_flat_list_of_images(images)

    def _prepare_input_images(
        self,
        images: ImageInput,
        do_convert_rgb: Optional[bool] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> List["torch.Tensor"]:
        """
        Prepare the input images for processing.

        This method is a compatibility shim for older transformers versions that
        don't have this method in BaseImageProcessorFast.
        """
        from functools import partial

        images = self._prepare_images_structure(images)

        # Check if parent has _process_image method
        if hasattr(super(), '_process_image'):
            process_image_fn = partial(
                super()._process_image,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )
            processed_images = []
            for image in images:
                processed_images.append(process_image_fn(image))
            return processed_images
        else:
            # Fallback: minimal processing for compatibility
            processed_images = []
            for image in images:
                # Convert to tensor if needed
                if not isinstance(image, torch.Tensor):
                    # Handle PIL images
                    if hasattr(image, 'convert'):
                        if do_convert_rgb:
                            image = image.convert('RGB')
                        # Convert PIL to tensor
                        image = F.to_tensor(image)
                    else:
                        # Handle numpy arrays
                        image = torch.from_numpy(image)
                        if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
                            image = image.permute(2, 0, 1)  # HWC -> CHW

                # Move to device if specified
                if device is not None:
                    image = image.to(device)

                processed_images.append(image)

            return processed_images

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        do_pad: bool,
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:

        image_sizes = [get_image_size(image, channel_dim=ChannelDimension.FIRST) for image in images]

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images)
        
        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes}, tensor_type=return_tensors
        )


    def preprocess(self, images: ImageInput, videos: VideoInput=None, **kwargs: Unpack[Eagle3_VLFastImageProcessorKwargs]) -> BatchFeature:
        # When fast image processing is not available, valid_kwargs may be dict without annotations
        valid_keys = getattr(self.valid_kwargs, '__annotations__', {}).keys() if hasattr(self.valid_kwargs, '__annotations__') else []
        if valid_keys:
            validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=valid_keys)
            # Set default kwargs from self. This ensures that if a kwarg is not provided
            # by the user, it gets its default value from the instance, or is set to None.
            for kwarg_name in valid_keys:
                kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        # Use get with defaults to handle missing kwargs in fallback mode
        do_convert_rgb = kwargs.pop("do_convert_rgb", getattr(self, 'do_convert_rgb', True))
        input_data_format = kwargs.pop("input_data_format", None)
        device = kwargs.pop("device", None)
        # Prepare input images
        if images is not None:
            images = self._prepare_input_images(
                images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
            )

        if videos is not None:
            videos = self._prepare_input_images(
                images=videos, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
            )

        # Update kwargs that need further processing before being validated
        if hasattr(self, '_further_process_kwargs'):
            kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        if hasattr(self, '_validate_preprocess_kwargs'):
            self._validate_preprocess_kwargs(**kwargs)

        # torch resize uses interpolation instead of resample
        resample = kwargs.pop("resample", getattr(self, 'resample', PILImageResampling.BICUBIC))
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        # Pop kwargs that are not needed in _preprocess (use pop with None default for safety)
        kwargs.pop("default_to_square", None)
        kwargs.pop("data_format", None)

        # In fallback mode, use simple preprocessing
        if not _FAST_IMAGE_PROCESSING_AVAILABLE:
            return self._preprocess_fallback(images if images is not None else videos, **kwargs)

        if images is not None:
            return self._preprocess(images, **kwargs)
        elif videos is not None:
            return self._preprocess(videos, **kwargs)

    def _preprocess_fallback(
        self,
        images: List["torch.Tensor"],
        **kwargs,
    ) -> BatchFeature:
        """Fallback preprocessing when fast image processing is not available."""
        # Get processing parameters with defaults
        do_rescale = kwargs.get("do_rescale", getattr(self, 'do_rescale', True))
        rescale_factor = kwargs.get("rescale_factor", 1.0 / 255.0)
        do_normalize = kwargs.get("do_normalize", getattr(self, 'do_normalize', True))
        image_mean = kwargs.get("image_mean", getattr(self, 'image_mean', IMAGENET_STANDARD_MEAN))
        image_std = kwargs.get("image_std", getattr(self, 'image_std', IMAGENET_STANDARD_STD))
        return_tensors = kwargs.get("return_tensors", None)

        image_sizes = []
        processed_images = []

        for image in images:
            if not isinstance(image, torch.Tensor):
                # Convert to tensor if needed
                if hasattr(image, 'convert'):  # PIL image
                    image = F.to_tensor(image)
                else:  # numpy
                    image = torch.from_numpy(image)
                    if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
                        image = image.permute(2, 0, 1)  # HWC -> CHW

            # Get image size (H, W)
            image_sizes.append((image.shape[-2], image.shape[-1]))

            # Rescale if needed
            if do_rescale:
                image = image.float() * rescale_factor

            # Normalize if needed
            if do_normalize and image_mean is not None and image_std is not None:
                mean = torch.tensor(image_mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
                std = torch.tensor(image_std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
                image = (image - mean) / std

            processed_images.append(image)

        # Stack images
        processed_images = torch.stack(processed_images)

        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes}, tensor_type=return_tensors
        )
    
__all__ = ["Eagle3_VLImageProcessorFast"]
