# Dataset Creation

## [Binary Dataset Creator](ExportHDF5.py)

Creates a binary hdf5 training dataset file based on a training and validation text file listing matching videos and labels.
Videos should have the same number of frames as the lines in the text file.
Text file should contain tab-delimited columns containing the annotations from each labeler.

## [Manual Dataset Editor](Cleanup_TrainSet.py)

Manually edits labels in an existing dataset with full control on the majority of the parameters.

## [Adding Unlabeled Videos To A Dataset](AddVideoToHDF5.py)

Script to add a video into the existing dataset. Defaults all the associated data (labels, mask, nlabelers) in as 0s.