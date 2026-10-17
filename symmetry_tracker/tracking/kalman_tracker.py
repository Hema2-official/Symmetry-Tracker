import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from symmetry_tracker.general_functionalities.io_utilities import GroupByFrame
from symmetry_tracker.general_functionalities.video_io import FrameSource
from symmetry_tracker.tracking.tracker_utilities import TrackingFrames, RecordCollector
from symmetry_tracker.tracking.post_processing import InterpolateGap


def EuclideanDistance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def NewKalmanFilter(Centroid):
  """The filter of a new track, initialized and updated with the centroid of its first object."""
  kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 states: x, y, dx, dy
  kf.F = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])  # state transition matrix
  kf.H = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])  # measurement function
  kf.R = 10  # measurement uncertainty
  kf.P *= 10  # initial state covariance
  kf.x[:2] = np.array(Centroid).reshape(-1, 2, 1)
  kf.update(np.array(Centroid))
  return kf


def KalmanTrackingFrames(FrameGroups, MaxCentroidDistance=20):
  """Kalman tracking over (Frame, rows) groups in frame order, one frame at a time."""
  NewTrackID = 1
  KalmanTracks = {}   # TrackID -> the filter of an active track
  LastRow = {}        # TrackID -> its latest row

  for Frame, Records in FrameGroups:
    if not Records:
      continue

    Matched = {}      # row index -> the TrackID it continues
    if KalmanTracks:
      TrackIDs = list(KalmanTracks)
      Predicted = []
      for TrackID in TrackIDs:
        KalmanTracks[TrackID].predict()
        Predicted.append(KalmanTracks[TrackID].x[:2][:, 0])
      DistanceMatrix = np.zeros((len(Records), len(TrackIDs)))
      for i, Record in enumerate(Records):
        for j, Prediction in enumerate(Predicted):
          DistanceMatrix[i, j] = EuclideanDistance(Record["Centroid"], Prediction)
      for i, j in zip(*linear_sum_assignment(DistanceMatrix)):
        if DistanceMatrix[i, j] <= MaxCentroidDistance:
          Matched[i] = TrackIDs[j]
      for TrackID in [TrackID for TrackID in KalmanTracks if TrackID not in Matched.values()]:
        del KalmanTracks[TrackID]
        del LastRow[TrackID]

    Continued = []
    for i, Record in enumerate(Records):
      if i in Matched:
        TrackID = Matched[i]
        KalmanTracks[TrackID].update(Record["Centroid"])
        Continued.append((LastRow[TrackID], Record))
      else:
        TrackID = NewTrackID
        NewTrackID += 1
        KalmanTracks[TrackID] = NewKalmanFilter(Record["Centroid"])
      Record["TrackID"] = TrackID
      LastRow[TrackID] = Record

    yield Frame, Records, Continued


def KalmanTracking(AnnotDF, MaxCentroidDistance = 20):
  print("Kalman Tracking")
  Rows = AnnotDF.to_dict(orient="records")
  for _ in KalmanTrackingFrames(GroupByFrame(Rows), MaxCentroidDistance):
    pass
  AnnotDF["TrackID"] = pd.Series([Row["TrackID"] for Row in Rows], index=AnnotDF.index, dtype=object)
  return AnnotDF


def SingleVideoKalmanTracking(VideoPath, AnnotPath, MaxCentroidDistance=20,
                              FaultyObjectRemoval = True, MinObjectPixelNumber = 20, MaxOverlapRatio = 0.5,
                              OutputPath=None):
  """
  - VideoPath: The path to video in stardard .png images format on which the tracking will be performed
  - AnnotPath: The path to a single annotation (segmentation) belonging to the video at VideoPath
  - MaxCentroidDistance: The maximal distance between estimated and measured centroid for them to be assigned as the same object
  - FaultyObjectRemoval: Boolean variable controlling whether to check for faulty segmentations and remove them
                        If True, objects with no valid centroids, too small object sizes or overlaps with other objects will be removed
  - MinObjectPixelNumber: Only active if FaultyObjectRemoval is True
                        Defines the minimal number of pixels in a Object istance for the instance to be recognised as valid
                        Object instances with PixelNumber<MinObjectPixelNumber will be simply deleted during initiation
  - MaxOverlapRatio:  Only active if FaultyObjectRemoval is True
                      The maximal overlap allowed between annotations in the original annotation.
                      Above MaxOverlapRatio, the area-wise smaller Object will be removed.
                      Not an important parameter if the segmentation is more or less a partitioning
  - OutputPath: Optional .json / .jsonl path. If given, the results are written there frame by frame and None
                is returned, so the memory needed does not depend on the video length. Otherwise the AnnotDF
                of the whole video is returned.
  """
  VideoShape = FrameSource(VideoPath, "GRAYSCALE", WindowSize=1).VideoShape   # only the shape of the video is needed
  Groups = TrackingFrames(AnnotPath, VideoShape, FaultyObjectRemoval, MinObjectPixelNumber, MaxOverlapRatio)

  print("Kalman Tracking")
  with RecordCollector(OutputPath) as Collector:
    for Frame, Records, Continued in KalmanTrackingFrames(Groups, MaxCentroidDistance):
      for LastRow, Row in Continued:
        Collector.AddInterpolated(InterpolateGap(LastRow, Row, Row["TrackID"]))
      Collector.AddOriginals(Frame, Records)
      Collector.Flush(Frame - 1)
    print(f"Successful interpolation of {Collector.NumInterpolated} object instances")
    return Collector.Finish()
