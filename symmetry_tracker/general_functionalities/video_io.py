"""
Frame access for videos stored as a folder of images.

- ListVideoFrames: the image files of a folder in frame order. Names are compared by their numeric
  runs (natural sort), so frame_10000.png comes after frame_2000.png whatever the zero padding.
- FrameSource: decodes frames on demand and keeps only a small window of recently used frames, so
  the memory needed does not depend on the length of the video.
"""
import functools
import os
import re

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_DIGITS = re.compile(r"(\d+)")


def FrameSortKey(FileName):
  """Sort key of a file name: numeric runs compare as integers, the rest as text."""
  Key = []
  for Part in _DIGITS.split(FileName):
    if Part.isdigit():
      Key.append((0, int(Part), ""))
    elif Part:
      Key.append((1, 0, Part))
  return Key


def ListVideoFrames(VideoPath):
  """The image files of VideoPath in frame order. Falls back to all files if none has an image extension."""
  Names = os.listdir(VideoPath)
  Images = [Name for Name in Names if Name.lower().endswith(IMAGE_EXTENSIONS)]
  return sorted(Images or Names, key=FrameSortKey)


class FrameSource:
  """
  The frames of a folder-of-images video, by index.

  - Color "GRAYSCALE": a frame is an (H, W) uint8 array.
  - Color "RGB": a frame is a (3, H, W) uint8 array (channels first, as the trackers consume it).
  Get keeps the WindowSize most recently used decoded frames.
  VideoShape is [NumFrames, H, W], as the trackers expect it.
  """

  def __init__(self, VideoPath, Color="GRAYSCALE", WindowSize=16):
    if not Color in ["GRAYSCALE", "RGB"]:
      raise Exception(f"{Color} is an invalid keyword for Color")
    self.VideoPath = VideoPath
    self.Color = Color
    self.Channels = 1 if Color == "GRAYSCALE" else 3
    self.Names = ListVideoFrames(VideoPath)
    self.NumFrames = len(self.Names)
    if self.NumFrames == 0:
      raise Exception(f"No frames found in {VideoPath}")
    self.Get = functools.lru_cache(maxsize=max(1, WindowSize))(self.Read)
    First = self.Get(0)
    self.Height, self.Width = First.shape[-2], First.shape[-1]
    self.VideoShape = [self.NumFrames, self.Height, self.Width]

  def Read(self, Index):
    """Decodes one frame from disk. Get(Index) is the cached version; what it returns must not be modified."""
    Path = os.path.join(self.VideoPath, self.Names[Index])
    if self.Color == "GRAYSCALE":
      Img = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
    else:
      Img = cv2.imread(Path)
      if Img is not None:
        Img = np.transpose(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB), (2, 0, 1))
    if Img is None:
      raise IOError(f"Could not read frame {Path}")
    return Img

  def LocalVideo(self, Frame, TimeKernelSize):
    """
    The 2*TimeKernelSize+1 frames around Frame as one (Channels*(2*TimeKernelSize+1), H, W) stack, the
    frames beyond the ends of the video replaced by the central frame: the kernel input LocalTracking
    builds per object.
    """
    K, C = TimeKernelSize, self.Channels
    Stack = np.repeat(self.Get(Frame).reshape(1, C, self.Height, self.Width), 2*K+1, axis=0)
    for dt in range(-K, K+1):
      if dt != 0 and 0 <= Frame+dt < self.NumFrames:
        Stack[dt+K] = self.Get(Frame+dt).reshape(C, self.Height, self.Width)
    return Stack.reshape((2*K+1)*C, self.Height, self.Width)
