"""Portable byte-representation identities for passive perception Artifacts."""

DEPTH_FRAME_MEDIA_TYPE_V1 = "application/x-npy"
DEPTH_FRAME_SCHEMA_ID_V1 = "dexteleop.perception.depth_image_meters.v1"
DEPTH_FRAME_ENCODING_V1 = "npy.float32.c_order.v1"
DEPTH_FRAME_COMPRESSION_V1 = "none"

RGB_FRAME_MEDIA_TYPE_V1 = "image/jpeg"
RGB_FRAME_SCHEMA_ID_V1 = "dexteleop.perception.rgb_frame.v1"
RGB_FRAME_ENCODING_V1 = "jpeg.rgb8.v1"
RGB_FRAME_COMPRESSION_V1 = "none"

__all__ = [
    "DEPTH_FRAME_COMPRESSION_V1",
    "DEPTH_FRAME_ENCODING_V1",
    "DEPTH_FRAME_MEDIA_TYPE_V1",
    "DEPTH_FRAME_SCHEMA_ID_V1",
    "RGB_FRAME_COMPRESSION_V1",
    "RGB_FRAME_ENCODING_V1",
    "RGB_FRAME_MEDIA_TYPE_V1",
    "RGB_FRAME_SCHEMA_ID_V1",
]
