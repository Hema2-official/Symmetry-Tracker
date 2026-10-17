import numpy as np
import cv2
import os
import ctypes
import torch
from concurrent.futures import ThreadPoolExecutor
from pycocotools import mask as coco_mask
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes

from symmetry_tracker.general_functionalities.video_io import ListVideoFrames
from symmetry_tracker.general_functionalities.io_utilities import RecordsToDataFrame

try:
  from IPython.display import display
  from symmetry_tracker.general_functionalities.misc_utilities import progress
except:
  pass

def PerformSegmentation(Predictor, VideoPath, Color = "GRAYSCALE", MinObjectSize = None):
  VideoFrames = ListVideoFrames(VideoPath)
  Outmasks = {}

  NumFrames = len(VideoFrames)
  print("Performing segmentation")
  try:
    ProgressBar = display(progress(0, NumFrames), display_id=True)
  except:
    pass
  for Frame in range(0,NumFrames):
    try:
      ProgressBar.update(progress(Frame, NumFrames))
    except:
      pass
    if Color == "GRAYSCALE":
      Img = cv2.imread(os.path.join(VideoPath,VideoFrames[Frame]), cv2.IMREAD_GRAYSCALE)
      Img = np.expand_dims(Img, axis=2)
    elif Color == "RGB":
      Img = cv2.cvtColor(cv2.imread(os.path.join(VideoPath, VideoFrames[Frame])), cv2.COLOR_BGR2RGB)
    else:
      raise Exception(f"{Color} is an invalid keyword for Color")

    Outputs = Predictor(Img)
    Outmask = (Outputs["instances"].pred_masks.to("cpu").numpy())
    ReducedOutmask = []
    for Segment in Outmask:
      if MinObjectSize is None or np.sum(Segment) >= MinObjectSize:
        ReducedOutmask.append(coco_mask.encode(np.asfortranarray(Segment)))
    Outmasks[Frame] = ReducedOutmask
  try:
    ProgressBar.update(progress(1, 1))
  except:
    pass
  print("Segmentation finished")
  return Outmasks

def MasksToAnnot(Outmasks):
  """The segmentation AnnotDF of the masks (linear time: built in one go)."""
  Records = []
  for Frame in Outmasks:
    NewObjectID = 1
    for SegmentationRLE in Outmasks[Frame]:
      FullObjectID = "{:04d}".format(Frame)+"{:04d}".format(NewObjectID)
      Records.append({"Frame": Frame, "ObjectID": FullObjectID, "SegmentationRLE": SegmentationRLE, "TrackID": None,
                      "Interpolated": False, "Class": None, "AncestorID": None})
      NewObjectID+=1

  return RecordsToDataFrame(Records, ["Frame", "ObjectID", "SegmentationRLE", "TrackID",
                                      "Interpolated", "Class", "AncestorID"])

try:
  MallocTrim = ctypes.CDLL("libc.so.6").malloc_trim
except (OSError, AttributeError):
  MallocTrim = None

def ParallelizePooler(Pooler, NumWorkers):
  """Makes a detectron2 ROIPooler pool the boxes of an image in chunks on NumWorkers threads."""
  if NumWorkers < 2:
    return
  OriginalForward = Pooler.forward
  NumChunks = 2*NumWorkers
  Pool = ThreadPoolExecutor(NumWorkers)

  def PoolChunk(Features, BoxTensor, Start, End):
    with torch.no_grad():
      return OriginalForward(Features, [Boxes(BoxTensor[Start:End])])

  def ChunkedForward(Features, BoxLists):
    if len(BoxLists) != 1 or len(BoxLists[0]) < NumChunks:
      return OriginalForward(Features, BoxLists)
    BoxTensor = BoxLists[0].tensor
    Bounds = [len(BoxTensor)*Chunk//NumChunks for Chunk in range(NumChunks+1)]
    Chunks = list(zip(Bounds[:-1], Bounds[1:]))
    Parts = [Pool.submit(PoolChunk, Features, BoxTensor, Start, End) for Start, End in Chunks]
    Output = None
    for (Start, End), Part in zip(Chunks, Parts):
      Pooled = Part.result()
      if Output is None:
        Output = torch.empty((len(BoxTensor),) + tuple(Pooled.shape[1:]), dtype=Pooled.dtype, device=Pooled.device)
      Output[Start:End].copy_(Pooled)
    if MallocTrim is not None:
      MallocTrim(0)
    return Output

  Pooler.forward = ChunkedForward

def SingleVideoSegmentation(VideoPath, ModelPath, ModelConfigPath, Device, Color = "GRAYSCALE", ScoreThreshold = 0.2, MinObjectSize = 20):
  """
  - VideoPath: The path to the video (in .png images format) to be segmented
  - ModelPath: The path to the model description file (.pth)
  - ModelConfigPath: The path to the model configuration file (.yaml)
  - Device: The device on which the segmentator should run (cpu or cuda:0)
  - Color: A keyword specific to the used model on the color encoding
           Available options: GRAYSCALE and RGB
  - ScoreThreshold: The acceptance threshold defining the object-ness of a given pixel
                    Lower values are more accepting
  - MinObjectSize: An optional parameter defining the minimal required object size in terms of pixel area
                  None is the default value
  """
  if not Color in ["GRAYSCALE", "RGB"]:
    raise Exception(f"{Color} is an invalid keyword for Color")

  cfg = get_cfg()
  cfg.merge_from_file(ModelConfigPath)
  cfg.MODEL.DEVICE = Device
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=ScoreThreshold
  cfg.MODEL.WEIGHTS = ModelPath
  cfg.INPUT.MIN_SIZE_TEST = 0
  cfg.INPUT.MAX_SIZE_TEST = 0
  Predictor = DefaultPredictor(cfg)
  BoxPooler = getattr(Predictor.model.roi_heads, "box_pooler", None)
  if BoxPooler is not None and torch.device(Device).type == "cpu":
    ParallelizePooler(BoxPooler, torch.get_num_threads())
  Outmasks = PerformSegmentation(Predictor, VideoPath, Color, MinObjectSize)
  AnnotDF = MasksToAnnot(Outmasks)

  return AnnotDF