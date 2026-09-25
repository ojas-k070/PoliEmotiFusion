# Future VideoMAE Training Pipeline

This document describes the intended training workflow for the future
video emotion model. It is a planning document only and does not claim
that model training has already occurred.

## 1. Dataset Preparation

The dataset preparation stage will collect and organize Indian political
video clips from approved and permitted sources.

Each metadata record should contain:

- video ID
- title
- source
- source URL
- category
- duration
- publication date
- local path
- emotion label
- dataset split

The `category` field describes the video context or event type, such as:

- speech
- debate
- rally
- protest
- strike
- confrontation
- parliamentary discussion

The `emotion_label` field is separate from the category and represents
the target emotion:

- anger
- joy
- sadness
- fear
- surprise
- disgust
- neutral

Newly collected videos may have no emotion label initially. They must be
manually or systematically annotated before being used for supervised
emotion-model training.

Local paths should remain portable and consistent.

Videos should not be unnecessarily committed to the GitHub repository.
Large datasets and model checkpoints should be stored separately from the
source-code repository when appropriate.

## 2. Annotation

Before training, each selected video clip should receive an emotion
annotation from the project's defined emotion taxonomy.

The annotation process should:

- use the same seven emotion labels throughout the dataset
- keep event/category labels separate from emotion labels
- document the annotation procedure
- identify ambiguous or low-confidence samples
- review politically sensitive edge cases carefully
- avoid assigning `neutral` simply because an emotion has not yet been
  annotated

Only samples with valid emotion annotations should enter the supervised
training pipeline.

## 3. Train/Validation/Test Split

Partition the annotated clips into:

- training set
- validation set
- test set

The current metadata schema supports:

- `train`
- `validation`
- `test`

The split should be created carefully to reduce data leakage.

In particular:

- avoid placing near-duplicate clips in different splits
- avoid using multiple clips from the same source segment across splits
- preserve class balance as much as practical
- keep the test set isolated until final evaluation

The exact split ratio will be decided after the dataset has been collected
and reviewed.

## 4. Frame Sampling

The current preprocessing pipeline supports temporal frame sampling.

The intended training configuration should:

- sample a fixed maximum number of frames per clip
- preserve temporal ordering
- avoid processing unnecessarily large numbers of frames
- record the sampling configuration used during training

The current implementation supports configurations such as:

- maximum frames: `32`
- sample rate: configurable
- maximum duration: configurable
- starting timestamp: configurable

The final number of frames may be adjusted after reviewing the dataset and
available computational resources.

## 5. Frame Preprocessing

Sampled frames will be prepared before being passed to the future model.

The current preprocessing pipeline supports:

- resizing frames to `224 × 224`
- conversion to a consistent NumPy array representation
- optional normalization of pixel values
- preservation of temporal frame order

The training pipeline should use the same preprocessing assumptions during
training and inference.

Any final preprocessing configuration should be stored with the training
configuration so that the trained model can be reproduced correctly.

## 6. VideoMAE Pretrained Checkpoint

The planned base model is:

`MCG-NJU/videomae-base`

The project intends to use a pretrained VideoMAE checkpoint rather than
training a video representation model from scratch.

The model name, local checkpoint path, and device configuration should be
controlled through the project's configuration/environment settings.

The current `VideoMAEAdapter` is intentionally a placeholder. It does not
download, load, or execute the pretrained model yet.

## 7. Model Adaptation

Once the dataset is prepared, the pretrained VideoMAE model will be adapted
for the project's seven-class emotion classification task.

The future model stage will:

1. initialize the pretrained VideoMAE checkpoint
2. configure the required classification head
3. set the number of output classes to seven
4. connect the project emotion labels to the model outputs
5. configure the training device
6. fine-tune the model using the annotated political-video dataset

The exact implementation will be added only after the dataset and training
requirements are finalized.

## 8. Fine-Tuning

The future fine-tuning process will:

- train the adapted VideoMAE model on the training set
- evaluate the model on the validation set after each epoch
- monitor training and validation performance
- save useful checkpoints
- review errors before finalizing the model

The training configuration should include parameters such as:

- learning rate
- batch size
- number of epochs
- optimizer
- weight decay
- frame sampling configuration
- input resolution
- random seed
- model checkpoint
- device

These values should be determined after the actual dataset and available
hardware have been evaluated.

## 9. Loss Function

The final loss function will depend on the annotation scheme and dataset
distribution.

For single-label emotion classification, the primary candidate is:

- cross-entropy loss

If significant class imbalance is observed, alternatives such as weighted
cross-entropy may be evaluated.

The final choice should be documented together with the training
configuration.

## 10. Optimizer

The exact optimizer will be selected during the real training setup.

Potential choices include:

- AdamW
- Adam

The final optimizer and its hyperparameters should be selected based on
the actual fine-tuning experiment rather than being assumed in advance.

## 11. Epochs and Training Strategy

The number of training epochs will be determined after reviewing:

- dataset size
- class distribution
- training stability
- validation performance
- available computational resources

Early stopping or another validation-based stopping strategy may be
considered if appropriate.

No final epoch count is claimed at the current foundation stage.

## 12. Evaluation

The trained model should be evaluated using multiple metrics rather than
accuracy alone.

Evaluation should include:

- validation loss
- test loss where appropriate
- overall accuracy
- per-class precision
- per-class recall
- per-class F1-score
- confusion matrix
- error analysis

Special attention should be given to politically sensitive and ambiguous
samples.

Examples of useful error-analysis cases include:

- emotionally neutral political speeches
- sarcastic or mocking commentary
- crowd reactions
- protests and confrontations
- rapidly changing emotional contexts
- clips containing multiple speakers
- clips where visual and spoken emotion differ

No model accuracy or performance number is claimed in this module
foundation.

## 13. Checkpoint Saving

The final training pipeline should save:

- model weights
- model configuration
- emotion-label mapping
- preprocessing configuration
- frame-sampling configuration
- dataset split information
- training metrics

The best checkpoint should be selected using a clearly documented
validation metric.

Model checkpoints should generally not be committed directly to the
source-code repository if they are large.

## 14. Temporal Emotion Analysis

The project is intended to analyze emotion over time rather than only
produce one prediction for the entire video.

The future pipeline should therefore:

1. divide the video into temporal clips or windows
2. obtain an emotion prediction for each window
3. associate predictions with timestamps
4. construct a temporal emotion timeline
5. identify meaningful changes in predicted emotion
6. aggregate the temporal predictions into a final distribution

The current project already provides the temporal-analysis structures
required to represent timestamped emotion predictions.

## 15. Inference

Once the trained model is connected, inference should follow this general
pipeline:

1. read and validate the input video
2. extract video metadata
3. sample frames according to the configured policy
4. preprocess the sampled frames
5. pass the processed frames to the trained VideoMAE model
6. obtain emotion probabilities
7. generate temporal predictions where applicable
8. aggregate predictions over the video
9. determine the dominant emotion
10. return the final emotion distribution and temporal analysis

The expected final result will contain information such as:

- dominant emotion
- confidence
- emotion probability distribution
- number of frames processed
- video duration
- temporal emotion results

## 16. Current Implementation Status

The current video module provides the foundation for the future training
pipeline.

Currently implemented:

- video metadata validation
- supported video-format validation
- frame sampling
- frame resizing
- frame preprocessing
- emotion aggregation
- temporal emotion timeline structures
- dataset metadata schema
- YouTube metadata collection scaffold
- VideoMAE adapter interface
- framework-neutral inference service

Not yet implemented:

- final annotated political-video dataset
- automated emotion annotation
- VideoMAE model loading
- VideoMAE fine-tuning
- trained emotion classification head
- final model checkpoint
- trained-model inference
- final model evaluation

Therefore, the current module is a **preparation and integration
foundation**, not a completed trained video emotion model.