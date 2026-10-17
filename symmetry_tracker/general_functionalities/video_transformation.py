import os

import cv2
import numpy as np

# Digits of the frame number in the file names. 8 digits keep frame_00010000.png after
# frame_00002000.png even for a plain lexicographic sort, which the 4-digit names broke from
# frame 10000 on. The readers sort numerically anyway (see video_io.ListVideoFrames).
FRAME_NAME_DIGITS = 8


def FrameFileName(Index):
  return f"frame_{Index:0{FRAME_NAME_DIGITS}d}.png"


def _IterPages(InputPath, Flags, ChunkSize):
  """Yields (page index, image) for every page of a multi-page image, reading ChunkSize pages at a time (OpenCV 4.5+)."""
  Start = 0
  while True:
    Ret, Pages = cv2.imreadmulti(InputPath, Start, ChunkSize, [], Flags)
    if not Ret or len(Pages) == 0:
      return
    for Offset, Page in enumerate(Pages):
      yield Start + Offset, Page
    if len(Pages) < ChunkSize:
      return
    Start += len(Pages)


def TransformVideoFromTIFF(InputPath, OutputPath, ChannelPeriodicity=1, StartPeriodicity=0, MaxSampleNum=None, ChunkSize=32):
  """
  Converts a video in TIF format into a series of PNG images stored in a specified folder.

  Parameters:
  - InputPath (str): The path to the TIF-formatted video file.
  - OutputPath (str): The destination folder where the PNG images will be saved.
  - ChannelPeriodicity (int, optional): Determines how frequently a non-fluorescent frame appears.
                                        If set to None, all frames are non-fluorescent channels.
  - StartPeriodicity (int, optional): Only relevant if ChannelPeriodicity is not None.
                                      Specifies the first frame that is non-fluorescent.
  - MaxSampleNum (int, optional): A debugging parameter that limits the number of frames loaded.
  - ChunkSize (int, optional): How many pages are read from the file at a time.

  Frames are named frame_<index>.png with FRAME_NAME_DIGITS digits.
  """
  SampleNum = 0
  for PageIndex, Page in _IterPages(InputPath, cv2.IMREAD_ANYCOLOR, ChunkSize):
    if ChannelPeriodicity is not None:
      if PageIndex < StartPeriodicity or (PageIndex - StartPeriodicity) % ChannelPeriodicity != 0:
        continue
    if MaxSampleNum is not None and SampleNum >= MaxSampleNum:
      break
    frame_path = os.path.join(OutputPath, FrameFileName(SampleNum))
    cv2.imwrite(frame_path, cv2.normalize(np.uint8(Page), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    SampleNum += 1
  print("Video transformed from "+InputPath)
