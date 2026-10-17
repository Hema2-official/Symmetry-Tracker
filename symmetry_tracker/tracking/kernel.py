"""
The tracking kernel: one forward pass of the time-symmetric model for one object.

Moved here unchanged from symmetry_tracker.py, so that the original whole-video functions and the
streaming implementation share exactly the same code (and produce exactly the same masks).
"""
import numpy as np
import torch
import torch.nn.functional as F


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
