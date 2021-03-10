# Inference Software

## [Example Inference Code](TestInfer.py)

Test code for running inferences using a single model.

## [General Inference Code](InferConsensusGrooming.py)

Code for running a consensus inference. Not all parameters have been exposed, but this script produces a raw output (pre-aggregatation for consensus) and a mean consensus result. Each frame is flipped and run across multiple models for consensus modalities.

## [Exporting a Merged Graph](ExportTFGraph.py)

Example export code for editing and merging multiple graphs for a more efficient consensus execution.

## [Inference Code for Exported Graphs](InferConsensusGrooming_TFExported.py)

Code for running a consensus inference using an exported graph. This code is highly specific to the exported graph and should be used as a model for how to run inference on an exported graph.

## [Plotting Activations](PlotActivations_Final.py)

Plots the activations of a network in addition to the final consensus onto a new video via a colormap for certainty. This includes the final selected point on the ROC curve (of 0.4811055 threshold with a 46 frame rolling average). [This file](PlotActivations.py) plots multiple consensus approaches without a rolling average.

## [Plotting Consensus](PlotConsensus.py)

Plots the consensus modalities onto a new video via a probability bar on the side (color = binary above/below threshold, size = relative probability).

## [Reading NPY Files](CompressNPY.py)

Generalized function for reading appended NPY output files.

## [Infer on HDF5 Dataset](InferValidationGrooming.py)

Example code for running an inference on the training/validation dataset.