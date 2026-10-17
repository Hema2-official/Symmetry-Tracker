import numpy as np

from symmetry_tracker.general_functionalities.io_utilities import LoadAnnotJSON, GroupByFrame


def EuclideanAncestors(FrameGroups, MaxCentroidDistance):
  """Returns {TrackID: mother TrackID or None}, one entry per track, in linear time."""
  Ancestor = {}
  FirstFrame = None
  for Frame, Records in FrameGroups:
    if FirstFrame is None:
      FirstFrame = Frame
    Present = {}
    for Record in Records:
      if Record["TrackID"] not in Present:
        Present[Record["TrackID"]] = Record["Centroid"]
    for child_tr_id, child_ctr in Present.items():
      if child_tr_id in Ancestor:
        continue
      mother_tr_id = None
      min_distance = None
      if Frame != FirstFrame:
        for mother_canditate_tr_id, mother_candidate_ctr in Present.items():
          if mother_canditate_tr_id != child_tr_id:
            distance = np.sqrt((mother_candidate_ctr[0] - child_ctr[0])**2 + (mother_candidate_ctr[1] - child_ctr[1])**2)
            if distance <= MaxCentroidDistance and (min_distance is None or distance < min_distance):
              min_distance = distance
              mother_tr_id = mother_canditate_tr_id
      Ancestor[child_tr_id] = mother_tr_id
  return Ancestor


def EuclideanInheritance(AnnotDF, MaxCentroidDistance):
  """Sets the AncestorID of every track that has a mother (see EuclideanAncestors); linear time."""
  Rows = AnnotDF.to_dict(orient="records")
  Ancestor = EuclideanAncestors(GroupByFrame(Rows), MaxCentroidDistance)
  Values = []
  for Row in Rows:
    Mother = Ancestor.get(Row["TrackID"])
    Values.append(Mother if Mother is not None else Row["AncestorID"])
  AnnotDF["AncestorID"] = Values
  return AnnotDF


def BuildInheritanceTree(VideoPath, AnnotPath, Method = "Euclidean", MaxCentroidDistance = 60):
  """
  - VideoPath: The path to the video (in .png images format)
  - AnnotPath: The path to the tracked annotation (json)
  - Method: The method of inheritance tree building. For now, only Euclidean is available
  - MaxCentoidDistance: The maximal allowed distance of centroids for object inheritance assignment
  """

  AnnotDF = LoadAnnotJSON(AnnotPath)
  if len(AnnotDF) < 1:
    raise Exception(f"Invalid TrackIDs for inheritance detection, run tracking beforehand")
  AnnotDF["AncestorID"] = None

  if Method == "Euclidean":
    AnnotDF = EuclideanInheritance(AnnotDF, MaxCentroidDistance)
  else:
    raise Exception(f"{Method} is not a valid keyword for Method")

  return AnnotDF
