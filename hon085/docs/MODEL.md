# Model documentation

## 1. Task

Classify one frame of a person into one of seven poses: `chair`, `cobra`, `dog`, `squat`, `standing`, `tree`, `warrior` (Warrior III). The classifier output drives the coach: pose name, confidence, hold time, repetitions, form score and active muscle groups.

## 2. Models

### 2.1 Pose landmarks: MediaPipe BlazePose (pre-trained)

BlazePose is a two-stage network. A detector finds the person and a landmark network regresses 33 body landmarks from the cropped image. In video the crop for the next frame is taken from the current landmarks, so the detector runs only when tracking is lost. Each landmark has image coordinates, a visibility score and a "world" position in metres relative to the hip centre. This project uses the world positions.

Three sizes are published: Lite, Full and Heavy. The training data was extracted with Heavy. The live app uses Full by default and lets the user switch.

### 2.2 Pose classifier (trained in this project)

| Property | Value |
|---|---|
| Input | 76 features computed from 33 world landmarks |
| Preprocessing | standardisation to zero mean and unit variance |
| Architecture | multilayer perceptron, hidden layers of 128 and 64 units, ReLU |
| Output | softmax over 7 classes |
| Training | Adam, cross-entropy loss, L2 penalty 0.001, up to 800 epochs |
| Parameters | 18,567 weights |
| Library | scikit-learn `MLPClassifier` |
| Decision | the top class if its probability is at least 0.60, otherwise "unknown" |

Four other algorithms are implemented behind the same interface for comparison: k-nearest neighbours, logistic regression, an RBF support vector machine and a random forest.

### 2.3 Features

All features come from `xray/features.py`.

1. **Centre and scale.** The hip centre is moved to the origin, the y axis is flipped to point up, and every coordinate is divided by the torso length (hip centre to shoulder centre). This removes position and body size.
2. **Yaw alignment.** The body is rotated about the vertical axis so that the left-to-right line through shoulders and hips lies on the x axis. This removes the direction the person faces.
3. **Coordinates.** x, y, z of 17 body landmarks (nose, shoulders, elbows, wrists, hips, knees, ankles, heels, toes): 51 values.
4. **Joint angles.** Inner angles at both elbows, shoulders, hips, knees and ankles: 10 values.
5. **Trunk lean.** Angle between the spine and the vertical: 1 value.
6. **Distances.** 14 distances between landmark pairs such as wrist to wrist, ankle to ankle and wrist to ankle.

The feature sets used in the experiments are nested: `raw` (untouched coordinates), `normalized` (step 1), `aligned` (steps 1 and 2), `full` (everything).

### 2.4 Coach

`xray/coach.py` smooths the class probabilities with an exponential filter (time constant 0.3 s) and changes the label only when a new class stays above 0.65 for 0.35 s. A label is dropped when its probability falls below 0.40. Entering a pose counts one repetition. `xray/form.py` checks a few alignment rules per pose, for example knee angle and trunk lean, and turns them into a score and hints. Poses taught by the user are compared with the joint angles recorded for them.

## 3. Data

| | Train | Test |
|---|---|---|
| Poses | 1634 | 661 |
| Subjects | 21 (5 rendered characters, 16 people) | 5 (1 rendered character, 4 people) |

| Class | Train | Test | Source |
|---|---|---|---|
| chair | 194 | 84 | yoga images |
| cobra | 195 | 116 | yoga images |
| dog | 199 | 90 | yoga images |
| squat | 275 | 84 | squat videos |
| standing | 374 | 86 | squat videos |
| tree | 200 | 96 | yoga images |
| warrior | 197 | 105 | yoga images |

- Yoga images: the test split of the original set is kept (character `guy3`). 19 of 1,495 images had no detectable person and were dropped.
- Squat videos: every sixth frame was processed. A frame is labelled `standing` when both knee angles are at least 142 degrees and `squat` when both are at most 118 degrees; frames in between are discarded. At most 12 frames per label and video are kept. People 17 to 20 form the test split.
- Augmentation: every training pose is also used mirrored (left and right swapped), which doubles the training rows.

## 4. Evaluation protocol

Model selection uses leave-one-subject-out cross-validation on the 21 training subjects. Accuracy and macro F1 are computed over the pooled out-of-fold predictions. The test subjects are used once, for the final model only. Splitting by subject matters because neighbouring video frames of one person are almost identical; a random split would leak them between train and test.

## 5. Results

### 5.1 Algorithms and feature sets (validation macro F1, unseen subjects)

| Algorithm | raw | normalized | aligned | full |
|---|---|---|---|---|
| k-nearest neighbours | 0.845 | 0.827 | 0.857 | 0.895 |
| Logistic regression | 0.761 | 0.748 | 0.851 | 0.898 |
| SVM (RBF) | 0.828 | 0.814 | 0.841 | 0.888 |
| Random forest | 0.828 | 0.807 | 0.855 | 0.873 |
| Neural network | 0.813 | 0.809 | 0.854 | 0.906 |

The full feature set is best for every algorithm. The neural network with full features is the selected model.

### 5.2 Network size and regularisation (validation)

| Hidden units | L2 penalty | Accuracy | Macro F1 |
|---|---|---|---|
| 32 | 0.0001 | 0.890 | 0.886 |
| 32 | 0.001 | 0.890 | 0.886 |
| 32 | 0.01 | 0.890 | 0.886 |
| 64 | 0.0001 | 0.907 | 0.906 |
| 64 | 0.001 | 0.908 | 0.907 |
| 64 | 0.01 | 0.909 | 0.908 |
| 128-64 | 0.0001 | 0.907 | 0.906 |
| 128-64 | 0.001 | 0.907 | 0.906 |
| 128-64 | 0.01 | 0.908 | 0.908 |
| 256-128 | 0.0001 | 0.910 | 0.909 |
| 256-128 | 0.001 | 0.910 | 0.909 |
| 256-128 | 0.01 | 0.909 | 0.908 |

All settings from 64 units upward are within half a point of each other, so the result does not depend on fine tuning. The 128-64 network was kept.

### 5.3 Held-out test

Accuracy 94.6 %, macro F1 0.945 on 661 poses.

| Class | Precision | Recall | F1 | Poses |
|---|---|---|---|---|
| chair | 1.000 | 1.000 | 1.000 | 84 |
| cobra | 0.943 | 1.000 | 0.971 | 116 |
| dog | 0.989 | 1.000 | 0.995 | 90 |
| squat | 0.933 | 0.988 | 0.960 | 84 |
| standing | 0.804 | 0.954 | 0.872 | 86 |
| tree | 1.000 | 1.000 | 1.000 | 96 |
| warrior | 0.974 | 0.705 | 0.818 | 105 |

Confusion matrix (rows: true class, columns: predicted class, same order):

| | chair | cobra | dog | squat | standing | tree | warrior |
|---|---|---|---|---|---|---|---|
| chair | 84 | 0 | 0 | 0 | 0 | 0 | 0 |
| cobra | 0 | 116 | 0 | 0 | 0 | 0 | 0 |
| dog | 0 | 0 | 90 | 0 | 0 | 0 | 0 |
| squat | 0 | 0 | 0 | 83 | 0 | 0 | 1 |
| standing | 0 | 1 | 1 | 1 | 82 | 0 | 1 |
| tree | 0 | 0 | 0 | 0 | 0 | 96 | 0 |
| warrior | 0 | 6 | 0 | 5 | 20 | 0 | 74 |

31 of the 36 errors are Warrior III poses, and 20 of them are labelled `standing`. The images checked from that group show the character from directly above. In that view BlazePose places the landmarks like those of an upright person, so the classifier answers `standing`.

With the 0.60 confidence threshold, 98.3 % of the test poses receive a label and 95.8 % of those labels are correct.

### 5.4 Mirror augmentation

| Training | Test accuracy | Mirrored test accuracy |
|---|---|---|
| without mirrored poses | 94.1 % | 78.4 % |
| with mirrored poses | 94.6 % | 94.6 % |

### 5.5 Robustness (test accuracy in %)

Gaussian noise added to every landmark, standard deviation in cm:

| Feature set | 0 | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|---|
| raw | 96.8 | 96.8 | 97.0 | 96.7 | 95.8 | 90.6 |
| normalized | 97.6 | 97.3 | 97.1 | 97.3 | 94.9 | 90.3 |
| aligned | 93.9 | 93.5 | 93.8 | 93.0 | 90.0 | 80.8 |
| full | 94.6 | 94.6 | 94.7 | 93.9 | 91.8 | 82.6 |

Test poses rotated about the vertical axis, in degrees:

| Feature set | 0 | 30 | 60 | 90 | 120 | 150 | 180 |
|---|---|---|---|---|---|---|---|
| raw | 96.8 | 96.2 | 84.7 | 63.2 | 54.8 | 70.7 | 75.3 |
| normalized | 97.6 | 95.6 | 80.8 | 63.2 | 58.9 | 62.5 | 70.7 |
| aligned | 93.9 | 93.9 | 93.9 | 93.9 | 93.9 | 93.9 | 93.9 |
| full | 94.6 | 94.6 | 94.6 | 94.6 | 94.6 | 94.6 | 94.6 |

On the unrotated test set the raw and normalized coordinates score higher than the full set, because the test character is filmed from the same camera positions as the training characters. They lose a third of their accuracy when the body is turned by 90 degrees. The aligned and full sets are unaffected by rotation, and the full set is the best on the 21 validation subjects, so it was chosen.

### 5.6 Speed (4 CPU cores)

| Algorithm | Training (s) | One prediction (ms) |
|---|---|---|
| k-nearest neighbours | 0.04 | 1.10 |
| Logistic regression | 2.89 | 0.60 |
| SVM (RBF) | 0.54 | 4.10 |
| Random forest | 1.07 | 76.55 |
| Neural network | 1.36 | 0.36 |

### 5.7 Videos from outside the dataset

Four exercise videos of other people (demo videos of the Exercise-Correction project by Ngo Hong Quoc Bao, MIT licence) were processed with BlazePose Full, a different model size than the training data. Share of frames per predicted label:

| Video | Frames | Predicted |
|---|---|---|
| bc | 170 | standing 95%, unknown 2%, tree 2%, warrior 1% |
| lunge | 160 | standing 64%, tree 27%, squat 7%, unknown 2% |
| plank | 115 | warrior 53%, dog 23%, cobra 22%, unknown 3% |
| squat | 129 | standing 51%, squat 47%, unknown 2% |

The curl video (a standing person) and the squat video are labelled as expected. The lunge and plank videos show the main limitation: there is no "other" class, so unfamiliar movements are given the nearest known pose with high confidence.

### 5.8 An approach that was tried and removed

To reject unknown poses, the distance to the nearest training poses in feature space was tested as a novelty score. It accepted 93 % of the test poses but also 96 % of the frames in the outside videos, so it could not separate known from unknown poses and was removed. The standing and squat classes from real recordings were added instead, which fixed the most common case, a person simply standing in front of the camera.

## 6. Capabilities and limitations

Capabilities
- Recognises seven poses from any horizontal viewing direction, for either body side.
- Runs in real time on a CPU; training takes about one second.
- New poses can be added from 20 or more samples without code.

Limitations
- Closed set: no reliable "none of these" answer.
- The yoga classes come from rendered characters; accuracy on photographs of people was not measured.
- The squat labels come from a knee-angle rule, so the classifier cannot be more correct than that rule for those two classes.
- One subject in the yoga test split. The yoga part of the test score has a wide uncertainty.
- Landmark errors propagate: occlusion, top-down views and loose clothing reduce accuracy.
- Depth from one camera is an estimate. A straight knee reads about 150 to 160 degrees, which the form rules take into account.
- Not a medical or diagnostic tool.
