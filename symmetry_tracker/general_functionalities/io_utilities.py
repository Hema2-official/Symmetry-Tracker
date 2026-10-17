"""
Annotation I/O.

The annotation of a video is a list of records (one per object instance). Besides the DataFrame API
(LoadAnnotJSON / ExportAnnotJSON, unchanged), records can be read one at a time (IterAnnotRecords)
and written one at a time (AnnotWriter), so the memory needed does not depend on the video length.
A .json file holds one JSON array, a .jsonl file one record per line. GroupByFrame and
GroupRecordsByFrame group rows by frame, for whole annotations and for streams respectively.
"""
import json
import re

import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask

from symmetry_tracker.general_functionalities.misc_utilities import CenterMass, BoundingBox

# Columns of the AnnotDF produced by LoadAnnotJSON
ANNOT_COLUMNS = ["Frame", "ObjectID", "SegmentationRLE", "Centroid", "SegBbox", "TrackID", "Interpolated",
                 "Class", "AncestorID"]
# Columns written by ExportAnnotJSON (what the front-end needs)
EXPORT_COLUMNS = ["Frame", "ObjectID", "SegmentationRLE", "TrackID", "Interpolated", "Class", "AncestorID"]

_SEPARATORS = re.compile(r"[\s,]*")
_WHITESPACE = re.compile(r"\s*")


def _IterJSONArray(File, ChunkSize=1 << 20):
  """
  Yields the elements of a JSON array from an open text file, decoding one element at a time, so the
  memory needed does not depend on the size of the file.
  """
  Decoder = json.JSONDecoder()
  Buffer = File.read(ChunkSize)
  Position = 0

  # the opening bracket
  while True:
    Position = _WHITESPACE.match(Buffer, Position).end()
    if Position < len(Buffer):
      break
    More = File.read(ChunkSize)
    if not More:
      return
    Buffer, Position = Buffer[Position:] + More, 0
  if Buffer[Position] != "[":
    raise ValueError("Expected a JSON array")
  Position += 1

  while True:
    # separators before the next element, or the closing bracket
    while True:
      Position = _SEPARATORS.match(Buffer, Position).end()
      if Position < len(Buffer):
        break
      More = File.read(ChunkSize)
      if not More:
        return
      Buffer, Position = Buffer[Position:] + More, 0
    if Buffer[Position] == "]":
      return
    # the element itself, reading more of the file until it is complete
    while True:
      try:
        Element, End = Decoder.raw_decode(Buffer, Position)
        break
      except json.JSONDecodeError:
        More = File.read(ChunkSize)
        if not More:
          raise
        Buffer, Position = Buffer[Position:] + More, 0
    yield Element
    Position = End


def IterAnnotJSON(AnnotPath):
  """
  Yields the raw records of an annotation file one by one.
  A .json file holds one JSON array (as written by ExportAnnotJSON), a .jsonl file one record per line.
  """
  with open(AnnotPath, "r") as File:
    if AnnotPath.endswith(".jsonl"):
      for Line in File:
        Line = Line.strip()
        if Line:
          yield json.loads(Line)
    else:
      for Element in _IterJSONArray(File):
        yield Element


def AnnotRecordFromJSON(Object):
  """
  One AnnotDF row (a dict) from one exported record, with the same stringification LoadAnnotJSON
  applied and the centroid and bounding box computed from the decoded mask.
  """
  IndividualSegImg = coco_mask.decode(Object["SegmentationRLE"])
  return {"Frame": Object["Frame"],
          "ObjectID": str(Object["ObjectID"]),
          "SegmentationRLE": Object["SegmentationRLE"],
          "Centroid": CenterMass(IndividualSegImg),
          "SegBbox": BoundingBox(IndividualSegImg),
          "TrackID": str(Object["TrackID"]),
          "Interpolated": bool(Object["Interpolated"]),
          "Class": str(Object["Class"]),
          "AncestorID": str(Object["AncestorID"])}


def IterAnnotRecords(AnnotPath):
  """Yields the AnnotDF rows (dicts) of an annotation file one by one."""
  for Object in IterAnnotJSON(AnnotPath):
    yield AnnotRecordFromJSON(Object)


def GroupByFrame(Records):
  """(Frame, rows) groups of rows given in any order, in frame order; the order of the rows within a frame is kept."""
  ByFrame = {}
  for Record in Records:
    ByFrame.setdefault(Record["Frame"], []).append(Record)
  return sorted(ByFrame.items())


def GroupRecordsByFrame(Records):
  """
  Yields (Frame, rows) for consecutive rows of the same frame, without reading ahead, so a stream of
  rows can be processed frame by frame. The rows must come in non-decreasing frame order.
  """
  CurrentFrame = None
  Group = []
  for Record in Records:
    Frame = Record["Frame"]
    if CurrentFrame is not None and Frame < CurrentFrame:
      raise ValueError(f"Annotation rows must be ordered by frame (frame {Frame} came after frame {CurrentFrame})")
    if Frame != CurrentFrame:
      if Group:
        yield CurrentFrame, Group
      CurrentFrame, Group = Frame, []
    Group.append(Record)
  if Group:
    yield CurrentFrame, Group


def RecordsToDataFrame(Records, Columns):
  """
  Builds a DataFrame from row dicts in one go. Every column has object dtype and holds the Python
  values as they are (the content the former row-by-row pd.concat produced); keys missing from a row
  become NaN.
  """
  Data = {}
  for Column in Columns:
    Data[Column] = pd.Series([Record.get(Column, np.nan) for Record in Records], dtype=object)
  return pd.DataFrame(Data, columns=Columns)


def LoadAnnotJSON(AnnotPath):
  """
  Loads the AnnotDF dataframe from a json
  Initiates centroid and bounding box based on the annotation
  (linear time: the rows are collected first and the DataFrame is built once)
  """
  return RecordsToDataFrame(list(IterAnnotRecords(AnnotPath)), ANNOT_COLUMNS)


def _JSONDefault(Value):
  if isinstance(Value, bytes):
    return Value.decode("ascii")
  if isinstance(Value, np.integer):
    return int(Value)
  if isinstance(Value, np.bool_):
    return bool(Value)
  if isinstance(Value, np.floating):
    return None if np.isnan(Value) else float(Value)
  if isinstance(Value, np.ndarray):
    return Value.tolist()
  raise TypeError(f"Cannot serialize {type(Value)}")


def ExportRecord(Record):
  """The exported view of one row: the EXPORT_COLUMNS, with missing or NaN values as None."""
  Exported = {}
  for Column in EXPORT_COLUMNS:
    Value = Record.get(Column)
    if isinstance(Value, float) and np.isnan(Value):
      Value = None
    Exported[Column] = Value
  return Exported


class AnnotWriter:
  """
  Writes exported rows one at a time, so results can be flushed while a video is still being
  processed. A .json path gives the JSON array layout of ExportAnnotJSON, a .jsonl path one record
  per line.
  """

  def __init__(self, SavePath):
    if SavePath.endswith(".jsonl"):
      self.Lines = True
    elif SavePath.endswith(".json"):
      self.Lines = False
    else:
      raise ValueError("SavePath must have a .json or .jsonl extension")
    self.File = open(SavePath, "w")
    self.Count = 0
    if not self.Lines:
      self.File.write("[")

  def Write(self, Record):
    Text = json.dumps(ExportRecord(Record), default=_JSONDefault, separators=(",", ":"))
    if self.Lines:
      self.File.write(Text + "\n")
    else:
      if self.Count:
        self.File.write(",")
      self.File.write(Text)
    self.Count += 1

  def Close(self):
    if self.File is None:
      return
    if not self.Lines:
      self.File.write("]")
    self.File.close()
    self.File = None

  def __enter__(self):
    return self

  def __exit__(self, *Args):
    self.Close()


def ExportAnnotJSON(AnnotDF, SavePath):
  """
  Saves the AnnotDF dataframe to a json (or jsonl)
  All unnecessary columns for the front-end are removed
  """
  if SavePath.endswith(".jsonl"):
    with AnnotWriter(SavePath) as Writer:
      for Record in AnnotDF[EXPORT_COLUMNS].to_dict(orient="records"):
        Writer.Write(Record)
  elif SavePath.endswith(".json"):
    AnnotDF_export = AnnotDF[EXPORT_COLUMNS].copy()
    AnnotDF_export.to_json(SavePath, orient="records")
  else:
    raise ValueError("SavePath must have a .json or .jsonl extension")
