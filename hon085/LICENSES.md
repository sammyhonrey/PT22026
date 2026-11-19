# Credits, references and licences

**See your Life X-ray** was designed and written by **Mohammad Kazim Honarjo** (Student ID 20243120085), 2026.

The code in `xray/`, `scripts/`, `tests/`, `run.py`, `web/js`, `web/css` and `web/index.html` is original work for this project.

## Third-party material

| What | Where | Author / owner | Licence |
|---|---|---|---|
| 3D anatomy meshes (895 structures) | `web/assets/anatomy/` | BodyParts3D, © The Database Center for Life Science | CC Attribution-Share Alike 2.1 Japan |
| Pose and hand landmark models | `web/models/` (downloaded) | Google, MediaPipe | Apache License 2.0 |
| MediaPipe Tasks Vision 0.10.21, web runtime | `web/vendor/mediapipe/` (downloaded) | Google | Apache License 2.0 |
| MediaPipe Python package | dependency | Google | Apache License 2.0 |
| three.js r170 | `web/vendor/three/` (downloaded) | three.js authors | MIT License |
| Droid Sans Bold font | `web/assets/fonts/` | Google / Ascender | Apache License 2.0 |
| scikit-learn, NumPy, joblib | dependencies | their authors | BSD 3-Clause |
| FastAPI, Uvicorn | dependencies | their authors | MIT, BSD 3-Clause |
| Pillow, Matplotlib | dependencies | their authors | MIT-CMU, PSF-based |

Licence texts are in `licenses/`.

## Data

| Dataset | Use | Source | Licence |
|---|---|---|---|
| Yoga poses image set | training and test data for five poses | TensorFlow pose classification tutorial, `download.tensorflow.org/data/pose_classification/yoga_poses.zip` | Apache License 2.0 |
| Dual-View Squat Movement Dataset | training and test data for standing and squat | `github.com/nihar787airobotics/squat_dataset_publication` | Terms not final at the time of use. Used for non-commercial coursework, cited, not redistributed. |
| Exercise-Correction demo videos | external check only | `github.com/NgoQuocBao1010/Exercise-Correction`, Ngo Hong Quoc Bao | MIT License |

No raw image or video is distributed with this project. `scripts/download_data.py` fetches the first two sources from their owners. `data/pose_landmarks.csv` contains only body landmark coordinates computed from them.

### Required attribution for the anatomy
> BodyParts3D, © The Database Center for Life Science licensed under CC Attribution-Share Alike 2.1 Japan.

The meshes were converted from the BodyParts3D 3.0 polygon data (STL copy at github.com/Kevin-Mattheus-Moerman/BodyParts3D), re-oriented, simplified to about 545,000 triangles, coloured and bound to 45 body segments by `scripts/build_anatomy.py`. Because the licence is Share-Alike, the adapted files in `web/assets/anatomy/` are shared under the same licence: https://creativecommons.org/licenses/by-sa/2.1/jp/deed.en. The attribution line is shown in the About tab and must stay visible.

## References
1. Bazarevsky V., Grishchenko I., Raveendran K., Zhu T., Zhang F., Grundmann M. BlazePose: On-device Real-time Body Pose Tracking. arXiv:2006.10204, 2020.
2. Lugaresi C. et al. MediaPipe: A Framework for Building Perception Pipelines. arXiv:1906.08172, 2019.
3. Zhang F. et al. MediaPipe Hands: On-device Real-time Hand Tracking. arXiv:2006.10214, 2020.
4. Pedregosa F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12, 2011, 2825-2830.
5. Mitsuhashi N. et al. BodyParts3D: 3D structure database for anatomical concepts. Nucleic Acids Research 37, 2009, D782-D785.
6. Casiez G., Roussel N., Vogel D. 1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems. CHI 2012.
7. TensorFlow. Human Pose Classification with MoveNet and TensorFlow Lite (tutorial and yoga pose data), 2021.
8. Kaminoff L., Matthews A. Yoga Anatomy, 2nd edition. Human Kinetics, 2012. (muscle groups of the yoga poses)

## Inspiration
The idea of a webcam "X-ray mirror" comes from the head-only web demo X-Ray Mirror by masterofvfx2070. That demo has no open-source licence and none of its code is used.

## Not a medical device
This software is an educational visual overlay and exercise aid. It is not a medical or diagnostic tool.
