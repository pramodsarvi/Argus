"""Optional DALI/NVDEC decode path (GPU box only). Sketch - verify args against your DALI version.

Use for the decode benchmark in the week-1 spike; production default is vlmpipe.frames (ffmpeg).
"""


def dali_sample(files: list[str], height: int, width: int, stride: int, device_id: int = 0):
    from nvidia.dali import fn, pipeline_def, types

    @pipeline_def(batch_size=1, num_threads=4, device_id=device_id)
    def pipe():
        # sequence_length=1 with stride yields single frames spaced `stride` apart.
        # NVDEC still decodes reference frames in between; this saves resize/transfer, not decode.
        v = fn.readers.video(device="gpu", filenames=files, sequence_length=1, stride=stride,
                             random_shuffle=False, name="reader")
        return fn.resize(v, resize_x=width, resize_y=height, interp_type=types.INTERP_CUBIC)

    p = pipe()
    p.build()
    n = p.epoch_size("reader")
    for _ in range(n):
        (out,) = p.run()
        yield out.as_cpu().as_array()[0, 0]   # HWC uint8
