"""
The tracking kernel: one forward pass of the time-symmetric model per object.

KernelTrackCentroid and KernelTrackBbox are the original functions, unchanged (the reference
LocalTracking uses them). TrackingKernel, used by the streaming implementation, computes the same
masks bit for bit with less work around the forward pass:

- the frame stack of a frame is converted once, through a 256-entry float32 table holding
  float32(float64(x)/255) (the values the original float64 chain produced), into a padded input
  buffer; per object only the marker channel is written. The original built three float64 copies
  of all 16 channels per object.
- the output is sigmoid(logits) > SegmentationConfidence compared in float32.
- no torch.cuda.empty_cache() per object.
"""
import numpy as np
import torch
import torch.nn.functional as F

# float32(float64(x)/255) for every uint8 value x: what the original float64 chain fed the model
PIXEL_VALUES = (np.arange(256, dtype=np.float64) / 255).astype(np.float32)


def ModelForward(Model, Inputs):
  """One forward pass of the model (a function of its own so that the benchmark can time it)."""
  with torch.no_grad():
    return Model(Inputs)


class TrackingKernel:
  """
  Runs the model for the objects of a frame, one forward pass per object, and returns the same
  (2K+1, H, W) boolean local tracks as KernelTrackCentroid / KernelTrackBbox.

  - Model, Device, Marker, SegmentationConfidence: as in SingleVideoSymmetryTracking
  - VideoShape: [NumFrames, H, W]; Channels: 1 (GRAYSCALE) or 3 (RGB)
  MarkerColumn is the record column the kernel marks the object with ("Centroid" or "SegBbox").
  """

  def __init__(self, Model, Device, VideoShape, Channels, TimeKernelSize, Marker, SegmentationConfidence):
    if not Marker in ["CENTROID", "BBOX"]:
      raise Exception(f"{Marker} is not an appropriate keyword for Marker")
    self.Device = torch.device(Device)
    self.Marker = Marker
    self.MarkerColumn = "Centroid" if Marker == "CENTROID" else "SegBbox"
    self.SegmentationConfidence = SegmentationConfidence
    self.Height, self.Width = VideoShape[1], VideoShape[2]
    PaddedHeight = self.Height + (16 - self.Height % 16) % 16
    PaddedWidth = self.Width + (16 - self.Width % 16) % 16
    self.NumFrameChannels = (2*TimeKernelSize+1)*Channels
    # the input of one object: the frame channels, then the marker channel; zero in the padding
    self.Inputs = torch.zeros((1, self.NumFrameChannels+1, PaddedHeight, PaddedWidth), dtype=torch.float32)
    self.Model = Model

  def LocalTracks(self, LocalVideo, Markers):
    """
    The local tracks of the objects of one frame. LocalVideo is the (NumFrameChannels, H, W) uint8
    stack of the frame (FrameSource.LocalVideo), Markers the marker value of every object (its
    centroid [CMy, CMx] or its bounding box [x0, y0, x1, y1], as in the records). Yields one
    (2K+1, PaddedH, PaddedW) bool array per marker, in order.
    """
    FrameChannels = self.Inputs[0, :self.NumFrameChannels, :self.Height, :self.Width]
    FrameChannels.copy_(torch.from_numpy(PIXEL_VALUES[LocalVideo]))
    # the marker is drawn on the unpadded frame, so slices beyond its edge are clipped as the original's were
    MarkerChannel = self.Inputs[0, self.NumFrameChannels, :self.Height, :self.Width]
    for MarkerValue in Markers:
      MarkerChannel.zero_()
      if self.Marker == "CENTROID":
        [CMy, CMx] = MarkerValue
        MarkerChannel[CMx-2:CMx+2, CMy-2:CMy+2] = 1.0
      else:
        [x0, y0, x1, y1] = MarkerValue
        MarkerChannel[x0:x1, y0:y1] = 1.0
      Logits = ModelForward(self.Model, self.Inputs.to(self.Device))
      yield (torch.sigmoid(Logits.cpu()) > self.SegmentationConfidence).numpy()[0]


# ---------------------------------------------------------------------------------------------------
# The original kernels, unchanged (moved here from symmetry_tracker.py); LocalTracking uses them.
# ---------------------------------------------------------------------------------------------------

def KernelTrackCentroid(LocalVideo, VideoShape, Model, Device, SegmentationConfidence, ObjectCenter):
  with torch.no_grad():
    inputs = LocalVideo
    CPImage = np.zeros([VideoShape[1],VideoShape[2]])
    [CMy,CMx]=ObjectCenter
    CPImage[CMx-2:CMx+2,CMy-2:CMy+2]=255
    inputs = np.append(inputs, [CPImage], axis=0)
    inputs = np.array(inputs, dtype=float)/255
    inputs = torch.Tensor(np.array(inputs))

    pad_h = (16 - inputs.shape[1] % 16) % 16
    pad_w = (16 - inputs.shape[2] % 16) % 16
    inputs = F.pad(inputs, (0, pad_w, 0, pad_h), mode='constant', value=0)

    inputs=torch.unsqueeze(inputs, dim=0)

    inputs=inputs.to(torch.device(Device))
    output = np.array(torch.sigmoid(Model(inputs).cpu()))
    output = output>SegmentationConfidence
    output = output*1.0
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)

    torch.cuda.empty_cache()

  return np.array(output[0], dtype = bool)


def KernelTrackBbox(LocalVideo, VideoShape, Model, Device, SegmentationConfidence, ObjectBbox):
  with torch.no_grad():
    inputs = LocalVideo
    BboxImg = np.zeros([VideoShape[1],VideoShape[2]])
    [x0, y0, x1, y1] = ObjectBbox
    BboxImg[x0:x1,y0:y1]=255
    inputs = np.append(inputs, [BboxImg], axis=0)
    inputs = np.array(inputs, dtype=float)/255
    inputs = torch.Tensor(np.array(inputs))

    pad_h = (16 - inputs.shape[1] % 16) % 16
    pad_w = (16 - inputs.shape[2] % 16) % 16
    inputs = F.pad(inputs, (0, pad_w, 0, pad_h), mode='constant', value=0)

    inputs=torch.unsqueeze(inputs, dim=0)

    inputs=inputs.to(torch.device(Device))
    output = np.array(torch.sigmoid(Model(inputs).cpu()))
    output = output>SegmentationConfidence
    output = output*1.0
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)

    torch.cuda.empty_cache()

  return np.array(output[0], dtype = bool)
