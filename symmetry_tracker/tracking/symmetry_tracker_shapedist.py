import time
import numpy as np
from pycocotools import mask as coco_mask
from scipy.optimize import linear_sum_assignment

from symmetry_tracker.general_functionalities.misc_utilities import CenterMass, shift_2d_replace, DecodeMultiRLE
from symmetry_tracker.general_functionalities.video_io import ListVideoFrames
from symmetry_tracker.tracking.tracker_metrics import TracksIOU
from symmetry_tracker.tracking.streaming_tracker import SymmetryTracking, MaximizeSimilarity

try:
  from IPython.display import display
  from symmetry_tracker.general_functionalities.misc_utilities import progress
except:
  pass

def TracksIOU_ShapeDistance(Array1, Array2, dt):
  """
  The 3D IoU of the channel pairs (dt+i, i), channel i of Array2 shifted by the centroid difference of
  the pair. A pair with an empty channel (no centroid) is compared unshifted.
  """
  if Array1.shape != Array2.shape:
    raise Exception(f"Dimensions of Array1 {Array1.shape} and Array2 {Array2.shape} must be the same")

  rolled_Array2 = np.zeros_like(Array2)
  for i, (seg1, seg2) in enumerate(zip(Array1[dt:], Array2[:-dt])):
    centroid1 = CenterMass(seg1)
    centroid2 = CenterMass(seg2)
    if None in centroid1 or None in centroid2:
      rolled_Array2[i] = seg2
    else:
      rolled_Array2[i] = shift_2d_replace(seg2, centroid1[0] - centroid2[0], centroid1[1] - centroid2[1])

  return TracksIOU(Array1, rolled_Array2, dt)


class ShapeTrack:
  """
  What ShapeIoU needs of one local track, computed once per object: the centroid (CenterMass) and the
  bounding box (y0, y1, x0, x1; None if empty) of every channel, and the track cropped to the box around
  all of its channels, with the offset (Y0, X0) of the crop in the frame.
  """
  __slots__ = ("Height", "Width", "Centroids", "Boxes", "Y0", "X0", "Crop")

  def __init__(self, LocalTrack, LocalTrackRLE):
    self.Height, self.Width = LocalTrack.shape[1], LocalTrack.shape[2]
    self.Centroids = [CenterMass(Channel) for Channel in LocalTrack]
    self.Boxes = []
    for x, y, w, h in coco_mask.toBbox(LocalTrackRLE):
      self.Boxes.append((int(y), int(y+h), int(x), int(x+w)) if w > 0 and h > 0 else None)
    Boxes = [Box for Box in self.Boxes if Box is not None]
    if Boxes:
      self.Y0, Y1 = min(Box[0] for Box in Boxes), max(Box[1] for Box in Boxes)
      self.X0, X1 = min(Box[2] for Box in Boxes), max(Box[3] for Box in Boxes)
    else:
      self.Y0, Y1, self.X0, X1 = 0, 0, 0, 0
    self.Crop = np.array(LocalTrack[:, self.Y0:Y1, self.X0:X1], dtype=bool)


def ShapeIoU(Track1, Areas1, Track2, dt):
  """
  TracksIOU_ShapeDistance(Array1, Array2, dt) from the ShapeTracks of the two local tracks (Areas1: the
  pixel counts of the channels of Array1). S_i is channel i of Array2 shifted and clipped to the frame;
  its intersection with A_i = Array1[dt+i] is counted inside the overlap of their boxes, and the union
  is sum_i |A_i| + sum_i |S_i| - intersection.
  """
  Height, Width = Track1.Height, Track1.Width
  Intersection = ShiftedArea = 0
  for i in range(len(Track1.Centroids) - dt):
    Box2 = Track2.Boxes[i]
    if Box2 is None:
      continue
    Centroid1, Centroid2 = Track1.Centroids[dt+i], Track2.Centroids[i]
    if None in Centroid1 or None in Centroid2:
      dx, dy = 0, 0
    else:
      dx, dy = Centroid1[0] - Centroid2[0], Centroid1[1] - Centroid2[1]
    y0, y1 = max(Box2[0]+dy, 0), min(Box2[1]+dy, Height)
    x0, x1 = max(Box2[2]+dx, 0), min(Box2[3]+dx, Width)
    if y0 >= y1 or x0 >= x1:
      continue
    # S_i on [y0, y1) x [x0, x1)
    Shifted = Track2.Crop[i, y0-dy-Track2.Y0:y1-dy-Track2.Y0, x0-dx-Track2.X0:x1-dx-Track2.X0]
    ShiftedArea += int(np.count_nonzero(Shifted))
    Box1 = Track1.Boxes[dt+i]
    if Box1 is None:
      continue
    iy0, iy1 = max(Box1[0], y0), min(Box1[1], y1)
    ix0, ix1 = max(Box1[2], x0), min(Box1[3], x1)
    if iy0 < iy1 and ix0 < ix1:
      Intersection += int(np.count_nonzero(np.logical_and(
          Track1.Crop[dt+i, iy0-Track1.Y0:iy1-Track1.Y0, ix0-Track1.X0:ix1-Track1.X0],
          Shifted[iy0-y0:iy1-y0, ix0-x0:ix1-x0])))
  Union = sum(Areas1[dt:]) + ShiftedArea - Intersection
  if Union == 0:
    return 0
  return Intersection / Union


class ShapeIoUSimilarity(MaximizeSimilarity):
  """
  The metric of GlobalAssignment_ShapeDistance: the IoU of the local tracks after aligning their
  centroids (TracksIOU_ShapeDistance), computed by ShapeIoU from the ShapeTrack that Prepare stores on
  every record.
  """

  def Prepare(self, Record, LocalTrack):
    Record["LocalTrackShape"] = ShapeTrack(LocalTrack, Record["LocalTrackRLE"])

  def Release(self, Record):
    Record["LocalTrackShape"] = None

  @staticmethod
  def ShapeOf(Record):
    """The ShapeTrack of a record, computed from its LocalTrackRLE if Prepare did not run on it."""
    Shape = Record.get("LocalTrackShape")
    if Shape is None:
      Shape = ShapeTrack(DecodeMultiRLE(Record["LocalTrackRLE"]), Record["LocalTrackRLE"])
      Record["LocalTrackShape"] = Shape
    return Shape

  def ScoreMatrix(self, T0, Tdt, dt):
    Matrix = np.zeros((len(T0), len(Tdt)))
    Shapes = [self.ShapeOf(Record) for Record in Tdt]
    for i, Record0 in enumerate(T0):
      Shape0, Areas0 = self.ShapeOf(Record0), Record0["LocalTrackAreas"]
      for j, ShapeDt in enumerate(Shapes):
        Matrix[i, j] = ShapeIoU(Shape0, Areas0, ShapeDt, dt)
    return Matrix


# Reference implementation (whole video in memory); the entry point below does not use it.
def GlobalAssignment_ShapeDistance(VideoPath, VideoShape, AnnotDF, TimeKernelSize, MinRequiredSimilarity=0.5, MaxTimeKernelShift=None):

  VideoFrames = ListVideoFrames(VideoPath)
  NumFrames = len(VideoFrames)

  if MaxTimeKernelShift is None:
    MaxTimeKernelShift=TimeKernelSize*2

  print("Global Assignment")

  try:
    ProgressBar = display(progress(0, len(AnnotDF)), display_id=True)
  except:
    pass

  AnnotDF[["PrevID", "NextID"]] = None
  for dt in range(1,MaxTimeKernelShift):
    for Frame in range(NumFrames-dt):

      # Similarity Matrix Calculation

      t0 = time.time()

      T0_IDs = AnnotDF.query("Frame == @Frame and NextID.isnull()", engine='python')["ObjectID"].tolist()
      Tdt_IDs = AnnotDF.query("Frame == @Frame+@dt and PrevID.isnull()", engine='python')["ObjectID"].tolist()

      SimilarityMatrix = np.zeros((len(T0_IDs),len(Tdt_IDs)))

      for i in range(len(T0_IDs)):
        T0_ID = T0_IDs[i]

        LTR0 = DecodeMultiRLE(AnnotDF.query("ObjectID == @T0_ID")["LocalTrackRLE"].iloc[0])
        for j in range(len(Tdt_IDs)):
          Tdt_ID = Tdt_IDs[j]

          #Removing bbox overlap based elimination as it is "unfair spatial information"

          LTRdt = DecodeMultiRLE(AnnotDF.query("ObjectID == @Tdt_ID")["LocalTrackRLE"].iloc[0])
          IOU = TracksIOU_ShapeDistance(LTR0, LTRdt, dt)
          SimilarityMatrix[i, j] = IOU

      SimilarityMatrix = SimilarityMatrix * (SimilarityMatrix >= MinRequiredSimilarity)

      # Hungarian Method based Assignment

      try:
        T0_assignedVals, Tdt_assignedVals = linear_sum_assignment(1-SimilarityMatrix)
      except:
        print(f"Error in linear_sum_assignment at Frame {Frame} to Frame {Frame+dt}")
        continue

      for k in range(len(T0_assignedVals)):
        if SimilarityMatrix[T0_assignedVals[k], Tdt_assignedVals[k]] >= MinRequiredSimilarity:
          T0_ID = T0_IDs[T0_assignedVals[k]]
          Tdt_ID = Tdt_IDs[Tdt_assignedVals[k]]
          AnnotDF.loc[AnnotDF.query("ObjectID == @T0_ID").index, "NextID"] = Tdt_ID
          AnnotDF.loc[AnnotDF.query("ObjectID == @Tdt_ID").index, "PrevID"] = T0_ID

      t_end = time.time()

      try:
        ProgressBar.update(progress(len(AnnotDF.query("not NextID.isnull()")), len(AnnotDF)))
      except:
        pass

  try:
    ProgressBar.update(progress(1, 1))
  except:
    pass

  return AnnotDF

def SingleVideoSymmetryTracking_ShapeDistance(VideoPath, ModelPath, Device, AnnotPath, TimeKernelSize,
                              Color = "GRAYSCALE", Marker = "CENTROID", SegmentationConfidence = 0.1,
                              MinRequiredSimilarity=0.5, MaxTimeKernelShift=None,
                              FaultyObjectRemoval = True, MinObjectPixelNumber=20, MaxOverlapRatio=0.5,
                              OutputPath=None, ModelOptimization="fast"):
  """
  - VideoPath: The path to video in stardard .png images format on which the tracking will be performed
  - ModelPath: The path to the pretrained model (the full model definition, not just the state dictionary)
  - Device: The device on which the segmentator should run (cpu or cuda:0)
  - AnnotPath: The path to a single annotation (segmentation) belonging to the video at VideoPath
  - TimeKernelSize: A constant parameter for the trained Tracker.
                    TimeKernelSize is the "radius" of the kernel, TimeKernelSize*2+1 is the "diamater" of the actual kernel.
  - Color: A keyword specific to the used model on the color encoding
           Available options: GRAYSCALE and RGB
  - Marker: A keyword specific to the used model on how the object to be tracked is marked
            Available options: CENTROID and BBOX
  - SegmentationConfidence: A number in [0,1] or defining the confidence threshold for the segmentation
                            Lower values are more allowing. Recommanded values are in the [0.1,0.9] range
  - MinRequiredSimilarity: The minimal required similarity based on IOU for two trackings to be possibly counted as belonging to the same Object
  - MaxTimeKernelShift: The maximal shift allowed between trackings to be recognised as belonging to the same Object
                        Minimal possible value: 1
                        Maximal possible value: 2*TimeKernelSize
                        The default None means maximal possible value
                        Usually None is recommended
                        Smaller values may result in trackings with more "breaks", but possibly fewer errors and slightly faster calculation
  - FaultyObjectRemoval: Boolean variable controlling whether to check for faulty segmentations and remove them
                        If True, objects with no valid centroids, too small object sizes or overlaps with other objects will be removed
  - MinObjectPixelNumber: Only active if FaultyObjectRemoval is True
                        Defines the minimal number of pixels in a Object istance for the instance to be recognised as valid
                        Object instances with PixelNumber<MinObjectPixelNumber will be simply deleted during initiation
  - MaxOverlapRatio:  Only active if FaultyObjectRemoval is True
                      The maximal overlap allowed between annotations in the original annotation.
                      Above MaxOverlapRatio, the area-wise smaller Object will be removed.
                      Not an important parameter if the segmentation is more or less a partitioning
  - OutputPath, ModelOptimization: as in SingleVideoSymmetryTracking
  """
  return SymmetryTracking(VideoPath, ModelPath, Device, AnnotPath, TimeKernelSize, Color, Marker, SegmentationConfidence,
                          ShapeIoUSimilarity(MinRequiredSimilarity), MaxTimeKernelShift,
                          FaultyObjectRemoval, MinObjectPixelNumber, MaxOverlapRatio, OutputPath, ModelOptimization)
