"""
Windowed (streaming) implementation of the tracking stage.

The frames of a video are processed in order. For frame g:
  1. the faulty objects of g are removed (TrackingFrames),
  2. the local track of every object of g is computed with the kernel (LocalTrackFrame),
  3. the global assignment pairs (g-dt, g), dt = 1..MaxTimeKernelShift-1, are solved (AssignFramePair),
  4. the objects of g get their TrackID (that of the object they were linked to, or a new one),
  5. the gaps of the new links are interpolated (InterpolateGap).
Only the local tracks of the last 2*TimeKernelSize-1 frames and 2*TimeKernelSize+1 decoded frames are
kept, so the memory does not grow with the length of the video, and the rows of a frame are final
(and can be written out) MaxTimeKernelShift-1 frames later.

Several similarity metrics can share step 2 (StreamingTrackingMulti): every metric gets its own copy of
the rows of a frame, sharing the local tracks, and performs steps 3-5 on them with its own window,
TrackIDs and output.
"""
import gc
from contextlib import ExitStack

import numpy as np
from pycocotools import mask as coco_mask
from scipy.optimize import linear_sum_assignment

from symmetry_tracker.general_functionalities.misc_utilities import EncodeMultiRLE, OuterBoundingBox, BoxOverlap
from symmetry_tracker.general_functionalities.video_io import FrameSource
from symmetry_tracker.tracking.kernel import TrackingKernel
from symmetry_tracker.tracking.post_processing import InterpolateGap
from symmetry_tracker.tracking.tracker_utilities import TrackingFrames, RecordCollector, LoadPretrainedModel


def TracksIOU_RLE(RLEs1, Areas1, RLEs2, Areas2, dt):
  """
  TracksIOU(DecodeMultiRLE(RLEs1), DecodeMultiRLE(RLEs2), dt) computed on the run-length encodings:
  the intersection and union pixel counts of the channel pairs (dt+i, i), summed, without decoding.
  Areas1 / Areas2 are the pixel counts of the channels.
  """
  if len(RLEs1) != len(RLEs2) or (len(RLEs1) > 0 and RLEs1[0]["size"] != RLEs2[0]["size"]):
    raise Exception("Dimensions of the two local tracks must be the same")
  Intersection = 0
  Union = 0
  for i in range(len(RLEs1) - dt):
    Inter = int(coco_mask.area(coco_mask.merge([RLEs1[dt + i], RLEs2[i]], intersect=True)))
    Intersection += Inter
    Union += Areas1[dt + i] + Areas2[i] - Inter
  if Union == 0:
    return 0
  return Intersection / Union


class SimilarityMetric:
  def Prepare(self, Record, LocalTrack):
    pass

  def Release(self, Record):
    pass


class MaximizeSimilarity(SimilarityMetric):
  """
  The matching rule of GlobalAssignment. A metric subclass fills the similarity matrix of two frames
  (ScoreMatrix); the pairs below MinRequiredSimilarity count as 0 for the Hungarian method, and an
  assigned pair is linked if its similarity is at least MinRequiredSimilarity.
  """

  def __init__(self, MinRequiredSimilarity):
    self.MinRequiredSimilarity = MinRequiredSimilarity

  def ScoreMatrix(self, T0, Tdt, dt):
    """The (len(T0), len(Tdt)) matrix of the scores of the candidate pairs, dt frames apart."""
    raise NotImplementedError

  def Cost(self, Matrix):
    return 1 - Matrix * (Matrix >= self.MinRequiredSimilarity)

  def Accept(self, Value):
    return Value >= self.MinRequiredSimilarity


class MinimizeDistance(SimilarityMetric):
  """
  The matching rule of GlobalAssignment_L2Distance. A metric subclass fills the distance matrix of two
  frames (ScoreMatrix), with MaxDistance for the pairs it does not score, and an assigned pair is linked
  if its distance is below MaxDistance.
  """

  def __init__(self, MaxDistance):
    self.MaxDistance = MaxDistance

  def ScoreMatrix(self, T0, Tdt, dt):
    raise NotImplementedError

  def Cost(self, Matrix):
    return Matrix

  def Accept(self, Value):
    return Value < self.MaxDistance


class TrackIoUSimilarity(MaximizeSimilarity):
  """The metric of GlobalAssignment: the 3D IoU of the local tracks of the pairs whose TrackBbox overlap."""

  def ScoreMatrix(self, T0, Tdt, dt):
    Matrix = np.zeros((len(T0), len(Tdt)))
    for i, Record0 in enumerate(T0):
      for j, RecordDt in enumerate(Tdt):
        if BoxOverlap(Record0["TrackBbox"], RecordDt["TrackBbox"]) != 0:
          Matrix[i, j] = TracksIOU_RLE(Record0["LocalTrackRLE"], Record0["LocalTrackAreas"],
                                       RecordDt["LocalTrackRLE"], RecordDt["LocalTrackAreas"], dt)
    return Matrix


def LocalTrackFrame(Frames, Frame, FrameRecords, Kernel, TimeKernelSize, Similarities):
  """
  The local track of every object of one frame: LocalTrackRLE, the pixel counts of its channels
  (LocalTrackAreas, from the encodings) and TrackBbox. The kernel runs once per object, as in
  LocalTracking. Returns one list of the rows per Similarity: shallow copies of FrameRecords made once
  these fields are set (so the copies share them), each with what its Similarity stores on it
  (SimilarityMetric.Prepare).
  """
  Copies = [[] for _ in Similarities]
  if not FrameRecords:
    return Copies
  LocalVideo = Frames.LocalVideo(Frame, TimeKernelSize)
  Markers = [Record[Kernel.MarkerColumn] for Record in FrameRecords]
  for Record, LocalTrack in zip(FrameRecords, Kernel.LocalTracks(LocalVideo, Markers)):
    Record["LocalTrackRLE"] = EncodeMultiRLE(LocalTrack)
    Record["LocalTrackAreas"] = [int(Area) for Area in coco_mask.area(Record["LocalTrackRLE"])]
    Record["TrackBbox"] = OuterBoundingBox(LocalTrack)
    for Similarity, Records in zip(Similarities, Copies):
      Records.append(dict(Record))
      Similarity.Prepare(Records[-1], LocalTrack)
  return Copies


def AssignFramePair(T0Records, TdtRecords, dt, Similarity):
  """
  Solves the assignment between the still unlinked objects of two frames, as one (Frame, Frame+dt)
  iteration of GlobalAssignment: scores the candidate pairs, runs the Hungarian method and links the
  accepted pairs (NextID / PrevID). Returns the new links as (Record0, RecordDt) pairs.
  """
  T0 = [Record for Record in T0Records if Record["NextID"] is None]
  Tdt = [Record for Record in TdtRecords if Record["PrevID"] is None]
  Links = []
  if not T0 or not Tdt:
    return Links
  Matrix = Similarity.ScoreMatrix(T0, Tdt, dt)
  try:
    Rows, Columns = linear_sum_assignment(Similarity.Cost(Matrix))
  except Exception:
    print(f"Error in linear_sum_assignment at Frame {T0[0]['Frame']} to Frame {Tdt[0]['Frame']}")
    return Links
  for i, j in zip(Rows, Columns):
    if Similarity.Accept(Matrix[i, j]):
      T0[i]["NextID"] = Tdt[j]["ObjectID"]
      Tdt[j]["PrevID"] = T0[i]["ObjectID"]
      Links.append((T0[i], Tdt[j]))
  return Links


class TrackingVariant:
  """
  One similarity metric of a tracking pass with its linking state: its rows of the frames in the window,
  the next new TrackID and the RecordCollector of its output. Performs steps 3-5 of the module docstring.
  """

  def __init__(self, Similarity, Collector):
    self.Similarity = Similarity
    self.Collector = Collector
    self.Window = {}      # Frame -> the rows of that frame with their local tracks, for the last MaxShift frames
    self.NextTrackID = 1

  def AddFrame(self, Frame, Records, MaxTimeKernelShift):
    """Links the rows of Frame to those of the frames before it, gives them their TrackID and interpolates the gaps."""
    self.Window[Frame] = Records

    NewLinks = []
    for dt in range(1, min(MaxTimeKernelShift, Frame+1)):
      NewLinks += AssignFramePair(self.Window.get(Frame-dt, []), Records, dt, self.Similarity)

    for Record0, RecordDt in NewLinks:
      RecordDt["TrackID"] = Record0["TrackID"]
      self.Collector.AddInterpolated(InterpolateGap(Record0, RecordDt, Record0["TrackID"]))
    for Record in Records:
      if Record["TrackID"] is None:
        Record["TrackID"] = self.NextTrackID
        self.NextTrackID += 1
    self.Collector.AddOriginals(Frame, Records)

  def Evict(self, Old):
    """Drops frame Old from the window with its local tracks and writes out the rows up to it, which are final."""
    for Record in self.Window.pop(Old, []):
      Record["LocalTrackRLE"] = None   # only needed while the frame is in the window
      self.Similarity.Release(Record)
    self.Collector.Flush(Old)


def StreamingTracking(Frames, FrameGroups, Kernel, TimeKernelSize, Similarity,
                      MaxTimeKernelShift=None, OutputPath=None, ProgressCallback=None):
  """
  Runs the whole tracking stage frame by frame (see the module docstring).

  - Frames: a FrameSource of the video
  - FrameGroups: the (Frame, tracking rows) groups of the annotation in frame order (TrackingFrames)
  - Kernel: the TrackingKernel of the model; TimeKernelSize: as in SingleVideoSymmetryTracking
  - Similarity: a MaximizeSimilarity or MinimizeDistance metric defining how objects of different frames are matched
  - MaxTimeKernelShift: as in SingleVideoSymmetryTracking
  - OutputPath: optional .json / .jsonl path; if given the rows are written there as they become final and None
                is returned, otherwise the AnnotDF of the whole video is returned (without the local tracks)
  - ProgressCallback: optional callable(FramesDone, NumFrames)
  """
  return StreamingTrackingMulti(Frames, FrameGroups, Kernel, TimeKernelSize, [(Similarity, OutputPath)],
                                MaxTimeKernelShift, ProgressCallback)[0]


def StreamingTrackingMulti(Frames, FrameGroups, Kernel, TimeKernelSize, Variants,
                           MaxTimeKernelShift=None, ProgressCallback=None):
  """
  StreamingTracking for several similarity metrics with one kernel pass: the local tracks are computed
  once per object, and every metric links its own copies of the rows (LocalTrackFrame) with its own
  window, TrackIDs and output, so its result is that of StreamingTracking with that metric alone.

  - Variants: a list of (Similarity, OutputPath) pairs, each as the Similarity and OutputPath of StreamingTracking
  - the other parameters as in StreamingTracking
  Returns the list of the results of the variants, in order.
  """
  if MaxTimeKernelShift is None:
    MaxTimeKernelShift=TimeKernelSize*2
  MaxShift = MaxTimeKernelShift - 1   # the largest dt of range(1, MaxTimeKernelShift)
  NumFrames = Frames.NumFrames
  Similarities = [Similarity for Similarity, _ in Variants]

  Groups = iter(FrameGroups)
  NextGroup = next(Groups, None)

  def Label(Variant):
    return f"{type(Variant.Similarity).__name__}: " if len(Variants) > 1 else ""

  print("Streaming tracking")
  with ExitStack() as Stack:
    Tracked = [TrackingVariant(Similarity, Stack.enter_context(RecordCollector(OutputPath)))
               for Similarity, OutputPath in Variants]
    for Frame in range(NumFrames):
      Records = []
      if NextGroup is not None and NextGroup[0] == Frame:
        Records = NextGroup[1]
        NextGroup = next(Groups, None)

      Copies = LocalTrackFrame(Frames, Frame, Records, Kernel, TimeKernelSize, Similarities)
      for Variant, VariantRecords in zip(Tracked, Copies):
        Variant.AddFrame(Frame, VariantRecords, MaxTimeKernelShift)
        Variant.Evict(Frame - MaxShift)

      if ProgressCallback:
        ProgressCallback(Frame+1, NumFrames)
      if (Frame+1) % 10 == 0 or Frame+1 == NumFrames:
        for Variant in Tracked:
          print(f"  {Label(Variant)}frame {Frame+1}/{NumFrames}: {Variant.Collector.NumOriginal} objects, "
                f"{Variant.NextTrackID-1} tracks, {Variant.Collector.NumInterpolated} interpolated")

    if NextGroup is not None:
      raise ValueError(f"The annotation has objects on frame {NextGroup[0]} but the video has only {NumFrames} frames")
    Results = []
    for Variant in Tracked:
      for Old in list(Variant.Window):
        Variant.Evict(Old)
      print(f"{Label(Variant)}Successful interpolation of {Variant.Collector.NumInterpolated} object instances")
      Results.append(Variant.Collector.Finish())
    return Results


def SymmetryTracking(VideoPath, ModelPath, Device, AnnotPath, TimeKernelSize, Color, Marker, SegmentationConfidence, Similarity,
                     MaxTimeKernelShift=None, FaultyObjectRemoval=True, MinObjectPixelNumber=20, MaxOverlapRatio=0.5,
                     OutputPath=None, ModelOptimization="fast"):
  """
  The common body of the SingleVideoSymmetryTracking* entry points: opens the video and the annotation,
  loads the model into a TrackingKernel and runs StreamingTracking with the given Similarity. The
  parameters are those of SingleVideoSymmetryTracking.
  """
  return SymmetryTrackingMulti(VideoPath, ModelPath, Device, AnnotPath, TimeKernelSize, Color, Marker,
                               SegmentationConfidence, [(Similarity, OutputPath)], MaxTimeKernelShift,
                               FaultyObjectRemoval, MinObjectPixelNumber, MaxOverlapRatio, ModelOptimization)[0]


def SymmetryTrackingMulti(VideoPath, ModelPath, Device, AnnotPath, TimeKernelSize, Color, Marker, SegmentationConfidence,
                          Variants, MaxTimeKernelShift=None, FaultyObjectRemoval=True, MinObjectPixelNumber=20,
                          MaxOverlapRatio=0.5, ModelOptimization="fast"):
  """
  SymmetryTracking for several similarity metrics on the same video with one kernel pass
  (StreamingTrackingMulti). Variants is a list of (Similarity, OutputPath) pairs; the metrics of the
  SingleVideoSymmetryTracking* entry points are TrackIoUSimilarity(MinRequiredSimilarity),
  CentroidL2Similarity(MaxCentroidDistance) (symmetry_tracker_l2dist) and
  ShapeIoUSimilarity(MinRequiredSimilarity) (symmetry_tracker_shapedist). Returns the list of the results,
  each that of the entry point of its metric; the other parameters are those of SingleVideoSymmetryTracking.
  """
  if not Marker in ["CENTROID", "BBOX"]:
    raise Exception(f"{Marker} is not an appropriate keyword for Marker")
  Frames = FrameSource(VideoPath, Color, WindowSize=2*TimeKernelSize+2)
  Groups = TrackingFrames(AnnotPath, Frames.VideoShape, FaultyObjectRemoval, MinObjectPixelNumber, MaxOverlapRatio)
  Model = LoadPretrainedModel(ModelPath, Device)
  Kernel = TrackingKernel(Model, Device, Frames.VideoShape, Frames.Channels, TimeKernelSize, Marker,
                          SegmentationConfidence, ModelOptimization)
  Results = StreamingTrackingMulti(Frames, Groups, Kernel, TimeKernelSize, Variants, MaxTimeKernelShift)
  del Kernel, Model
  gc.collect()
  return Results
