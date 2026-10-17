import numpy as np
import torch
from pycocotools import mask as coco_mask

from symmetry_tracker.general_functionalities.misc_utilities import BoxOverlap
from symmetry_tracker.general_functionalities.io_utilities import (RecordsToDataFrame, IterAnnotRecords, GroupByFrame,
                                                                   GroupRecordsByFrame, AnnotWriter)
from symmetry_tracker.tracking.tracker_metrics import AnnotationOverlap

# Columns of the tracking AnnotDF
TRACKING_COLUMNS = ["Frame", "ObjectID", "SegmentationRLE", "LocalTrackRLE", "Centroid", "SegBbox", "TrackBbox",
                    "PrevID", "NextID", "TrackID", "Interpolated", "Class", "AncestorID"]


def TrackingRecord(AnnotRecord):
  """One tracking row (a dict) from one loaded annotation row, the tracking columns initialized as TransformToTrackingAnnot does."""
  return {"Frame": AnnotRecord["Frame"],
          "ObjectID": AnnotRecord["ObjectID"],
          "SegmentationRLE": AnnotRecord["SegmentationRLE"],
          "LocalTrackRLE": None,
          "Centroid": AnnotRecord["Centroid"],
          "SegBbox": AnnotRecord["SegBbox"],
          "TrackBbox": None,
          "PrevID": None,
          "NextID": None,
          "TrackID": None,
          "Interpolated": False,
          "Class": AnnotRecord["Class"],
          "AncestorID": AnnotRecord["AncestorID"]}


def TransformToTrackingAnnot(AnnotDF):
  """Properely initializes the None columns (LocalTrackRLE, TrackBbox, PrevID, NextID, TrackID) of the AnnotDF."""
  Records = [TrackingRecord(Row) for Row in AnnotDF.to_dict(orient="records")]
  return RecordsToDataFrame(Records, TRACKING_COLUMNS)


def FaultyObjectIDs(FrameRecords, VideoShape, MinObjectPixelNumber, MaxOverlapRatio):
  """
  The ObjectIDs of the rows of one frame that RemoveFaultyObjects drops: too small, no centroid or a
  centroid on the border, or covered by a larger overlapping object by more than MaxOverlapRatio.
  Every mask is decoded at most once.
  """
  Decoded = {}

  def Seg(Record):
    Mask = Decoded.get(Record["ObjectID"])
    if Mask is None:
      Mask = coco_mask.decode(Record["SegmentationRLE"])
      Decoded[Record["ObjectID"]] = Mask
    return Mask

  Faulty = []
  for Record in FrameRecords:
    ObjectID = Record["ObjectID"]
    ObjectBbox = Record["SegBbox"]
    ObjectSeg = Seg(Record)
    Size = np.sum(ObjectSeg)
    [Cx,Cy] = Record["Centroid"]

    HasBetterCoverage = False
    for Record2 in FrameRecords:
      if ObjectID != Record2["ObjectID"]:
        if BoxOverlap(ObjectBbox, Record2["SegBbox"]) > 0:
          Object2Seg = Seg(Record2)
          if Size <= np.sum(Object2Seg) and AnnotationOverlap(ObjectSeg, Object2Seg) > MaxOverlapRatio:
            HasBetterCoverage = True

    if Size<MinObjectPixelNumber or [Cx, Cy]==[None,None] or Cx<=0 or Cy<=0 or Cx>=VideoShape[2]-1 or Cy>=VideoShape[1]-1 or HasBetterCoverage:
      Faulty.append(ObjectID)
  return Faulty


def RemoveFaultyObjects(AnnotDF, VideoShape, MinObjectPixelNumber, MaxOverlapRatio):
  """
  Removes the objects from the annotation, which
  - have too small size based on self.MinObjectPixelNumber
  - have no defineable center
  """
  print("Removing faulty object instances")
  Faulty = []
  for Frame, Rows in GroupByFrame(AnnotDF.to_dict(orient="records")):
    Faulty += FaultyObjectIDs(Rows, VideoShape, MinObjectPixelNumber, MaxOverlapRatio)
  if Faulty:
    AnnotDF = AnnotDF[~AnnotDF["ObjectID"].isin(Faulty)]
  print(f"Number of removed faulty object instances: {len(Faulty)}")
  return AnnotDF


def TrackingFrames(AnnotPath, VideoShape, FaultyObjectRemoval=True, MinObjectPixelNumber=20, MaxOverlapRatio=0.5):
  """
  The annotation at AnnotPath as (Frame, tracking rows) groups in frame order, read one frame at a time,
  with the faulty objects of every frame removed (FaultyObjectIDs) if FaultyObjectRemoval is set.
  The common first step of the tracking entry points.
  """
  NumFaulty = 0
  Records = (TrackingRecord(Record) for Record in IterAnnotRecords(AnnotPath))
  for Frame, Rows in GroupRecordsByFrame(Records):
    if FaultyObjectRemoval:
      Faulty = set(FaultyObjectIDs(Rows, VideoShape, MinObjectPixelNumber, MaxOverlapRatio))
      NumFaulty += len(Faulty)
      Rows = [Row for Row in Rows if Row["ObjectID"] not in Faulty]
    yield Frame, Rows
  if FaultyObjectRemoval:
    print(f"Number of removed faulty object instances: {NumFaulty}")


class RecordCollector:
  """
  Gathers the result rows of a tracking run. Eliminates linear memory usage and DF overhead.
  Use it as a context manager so the file is closed on errors too.
  """

  def __init__(self, OutputPath=None):
    self.Writer = AnnotWriter(OutputPath) if OutputPath is not None else None
    self.Originals = []             # DataFrame mode
    self.Interpolated = []
    self.Pending = {}               # writer mode: Frame -> rows of that frame not written yet
    self.PendingInterpolated = {}
    self.NumOriginal = 0
    self.NumInterpolated = 0

  def __enter__(self):
    return self

  def __exit__(self, *Args):
    self.Close()

  def Close(self):
    if self.Writer is not None:
      self.Writer.Close()

  def AddOriginals(self, Frame, Records):
    self.NumOriginal += len(Records)
    if self.Writer is None:
      self.Originals.extend(Records)
    elif Records:
      self.Pending.setdefault(Frame, []).extend(Records)

  def AddInterpolated(self, Records):
    self.NumInterpolated += len(Records)
    if self.Writer is None:
      self.Interpolated.extend(Records)
    else:
      for Record in Records:
        self.PendingInterpolated.setdefault(Record["Frame"], []).append(Record)

  def Flush(self, UpToFrame):
    """Writer mode: writes every pending frame up to and including UpToFrame."""
    if self.Writer is None:
      return
    for Frame in sorted(set(self.Pending) | set(self.PendingInterpolated)):
      if Frame > UpToFrame:
        break
      for Record in self.Pending.pop(Frame, []):
        self.Writer.Write(Record)
      for Record in sorted(self.PendingInterpolated.pop(Frame, []), key=lambda R: R["TrackID"]):
        self.Writer.Write(Record)

  def Finish(self, Columns=TRACKING_COLUMNS):
    if self.Writer is not None:
      self.Flush(float("inf"))
      self.Close()
      return None
    self.Interpolated.sort(key=lambda R: (R["TrackID"], R["Frame"]))
    return RecordsToDataFrame(self.Originals + self.Interpolated, Columns)


def LoadPretrainedModel(ModelPath, Device):
  Model = torch.load(ModelPath, map_location = torch.device(Device))
  Model.eval()
  print("Model successfully loaded from:")
  print(ModelPath)
  return Model
