# TASK 4: Video Processing & Multi-Object Identity Tracking

Task 4 addresses continuous multi-object egg detection, identity association, motion modeling, and occlusion recovery in video sequences under high-speed spinning motion and severe human hand interactions.

---

## 1. Input Video Properties & Challenges

- **Input Footage:** `Raw and Boiled Egg Spinning - Cool Science Experiment | Educational Videos by Mocomi`
- **Native Resolution:** $1920 \times 1080$ px
- **Frame Rate:** $24.0$ frames per second
- **Total Duration:** $60.04$ seconds ($1,441$ video frames)
- **Primary Challenges:**
  1. **High Angular Velocity:** Eggs spin rapidly, altering aspect ratio, boundary gradients, and surface texture.
  2. **Severe Occlusion:** The presenter repeatedly interacts with, grasps, and stops the spinning eggs.
  3. **Visual Reflections:** Polished reflective table surface creates dynamic specular mirror images.
  4. **Appearance Homogeneity:** Raw and boiled eggs exhibit near-identical exterior color profiles.

---

## 2. Tracker Evolution: From Classical Failures to Foundation SORT

Multiple tracker paradigms were designed and tested across versions V1 through V8:

| Version | Approach | Observed Outcome | Failure Mode & Diagnosis | Verdict |
| :---: | :--- | :--- | :--- | :---: |
| **V1** | Hough Circles + Local Search | Initial tracking OK; lost eggs during high-speed spinning | Centroid jumped onto experimenter's hand during interaction | Prototype |
| **V2** | E1 Candidates + Joint 2-Track Assignment | Rigid gates rejected spinning eggs; tracks dropped to LOST | Deformation during spinning violated rigid aspect gates | ❌ Rejected |
| **V3** | Forced 2 Persistent IDs | Both tracks collapsed onto a single egg during close proximity | Forcing persistent IDs caused clutter hijacking during occlusion | ❌ Rejected |
| **V4** | Conservative Gate Tracker | Tracks frequently entered LOST state; failed reacquisition | Overly conservative gating caused premature track termination | ❌ Rejected |
| **V5** | Hybrid Confidence Tracker | Tracker drifted onto moving hands across the table | Brightness weighting latched onto pale skin tones | ❌ Rejected |
| **V6** | Lucas-Kanade Optical Flow + E1 | Optical flow drifted when egg and hand merged into one component | Upstream color thresholding cannot separate hands from eggs | ❌ Rejected |
| **V7** | Multi-Scale Template Matching | Egg 1 tracked; Egg 2 lost entirely during spin | High spin speed destroyed template appearance correlation | ❌ Rejected |
| **V8** | Protected Template + Global Reacquisition | Reacquired onto human hand after occlusion | Normalized cross-correlation scored high on human palm | ❌ Rejected |
| **FINAL** | **Grounding DINO + SORT** | **1,441 frames processed; 0 hand hijackings; stable tracks** | Occasional ID fragmentation across prolonged occlusions (>12 frames) | ✅ **Selected Winner** |

---

## 3. The Final Architecture: Grounding DINO + SORT

$$\text{Video Frame } \xrightarrow[\text{Stride } = 3]{\text{Open-Vocabulary DINO}} \text{Detections } \mathcal{D}_t \xrightarrow[\text{Cost Matrix } C]{\text{IoU Association}} \text{Hungarian Matching} \xrightarrow{} \text{Kalman State Update } \mathbf{x}_{t|t}$$

### Key Components
1. **Per-Frame Detection:** Grounding DINO Tiny prompt `"egg."` with detection threshold $0.20$.
2. **Detection Stride & Interpolation:** Evaluated every 3 frames for real-time throughput ($4\times$ speedup via 960px width downscaling); Kalman filter state prediction interpolates intermediate steps.
3. **State Representation:** 7-dimensional continuous state vector:
   $$\mathbf{x} = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$$
   representing center coordinates $(u, v)$, bounding box scale $s$, aspect ratio $r$, and respective velocities.
4. **Data Association:** Bounding box IoU distance matrix solved via the Hungarian / Kuhn-Munkres algorithm.
5. **Occlusion Management:** Tracks maintain state predictions for up to $12$ missed frames before being dropped, enabling recovery when eggs re-emerge from hand occlusion.

---

## 4. Final Tracking Operational Telemetry

- **Total Video Frames:** 1,441 / 1,441 ($100\%$ full-video coverage)
- **Track IDs Assigned:** 19 unique tracks across the 60-second video
- **Sum of Matched Frames:** 618 track-frames
- **Sum of Predicted Frames:** 1,435 track-frames
- **Dominant Track Lifespans:** Track 3 (22.25s), Track 4 (12.13s), Track 11 (11.88s)
- **Hand Hijacking Events:** **0** (Zero false hand tracks)

---

## 5. Directory Organization

```text
TASK 4/
├── README.md
├── scripts/
│   ├── 35_task4_gdino_sort.py           # Final production Grounding DINO + SORT pipeline
│   ├── tracker_canonical.py             # Modular tracker architecture
│   ├── 31_task4_candidate_analysis.py   # Initial candidate video frame exploration
│   ├── 32_task4_tracking_prototype.py   # V1 prototype
│   ├── 33_task4_tracking_v2.py          # V2 tracking trial
│   ├── 34_task4_tracking_v3-v8.py       # V3-V8 classical & template tracking trials
│   └── 35_task4_tracking_v2-v3.py
├── tracking/
│   └── task4_final_tracking_metrics.csv # Frame-by-frame bounding box & identity log
├── reports/                             # CSV telemetry and operational logs
├── visualizations/                      # Extracted frame overlays and debugging snapshots
└── demo/                                # Demo video access details & external links
```

---

## 6. Access to Rendered Video Demonstrations

Full 1080p rendered tracking videos exceed GitHub file-size limits. They are publicly accessible:
- **Full Video Demonstration (Google Drive):** [https://drive.google.com/file/d/1eiu521OtXV-wBelTvI9uoibKpDdwo55m/view?usp=sharing](https://drive.google.com/file/d/1eiu521OtXV-wBelTvI9uoibKpDdwo55m/view?usp=sharing)
- **To reproduce tracking locally:**
  ```bash
  python "TASK 4/scripts/35_task4_gdino_sort.py" \
    --video-path "path/to/video.webm" \
    --output-dir "TASK 4/tracking"
  ```
