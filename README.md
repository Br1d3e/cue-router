# CueRouter

CueRouter is a pool (pocket billiards) practice assistant focused on Chinese 8-ball. Give it a photo of the table and it can suggest which ball to pot, which pocket to use, and where to leave the cue ball for the next shot. When you already have a shot in mind, it can search for the cue direction, speed, and tip contact needed to play it.

## Why I built it

I enjoy playing pool and wanted to get better at cue-ball positioning. CueRouter started as a basic searcher for a question I kept asking at the table: how do I pot this ball and leave the cue ball where I want it? That searcher became the executor. The planner came later to help decide where the cue ball should finish in the first place.

## How it works

### Table vision: from photo to layout

[Table vision](table_vision/) uses a fine-tuned YOLOv8 detector to identify balls and pockets. It uses the pockets to calibrate a homography that maps ball locations from image pixels into table coordinates. The starting weights came from the billiards detector in [motern88/Det-SAM2](https://github.com/motern88/Det-SAM2), specifically its [pretrained YOLOv8 checkpoint](https://huggingface.co/ATA-space/Det-SAM2/blob/main/det_weights/train_referee12_960.pt). CueRouter uses the detector, not the SAM2 video-tracking pipeline.

### Executor: how to play a chosen shot

The [executor](cue_router/executor/) receives a [`SearchRequest`](cue_router/core/request.py) with the current table state, target ball, target pocket, cue-ball target region, and route type (direct or one-rail bank). Its job is to find the best-scoring [`ShotAction(phi, V0, a, b)`](cue_router/executor/action.py) it can: `phi` is the aiming direction, `V0` is the initial cue speed, and `a` and `b` are the horizontal and vertical tip offsets that control side spin and follow or draw.

The goal is to pot the target ball in the requested pocket with a legal first contact and no scratch, while leaving the cue ball in the desired region. `pooltool` simulates each candidate. Search can start from a grid or stochastic samples and refine the strongest candidates with local pattern search. A separate CEM-inspired elite search mixes fresh samples with perturbations of high-scoring actions. The executor reranks candidates for robustness to small changes in aim, speed, and tip contact.

### Planner: which shot to play

The [planner](cue_router/planner/) ranks shot options from the detected layout and the set of legal target balls. It estimates potting ease using physics-based heuristics such as line of sight, cut angle, shot distances, and cueing room. For each possible pot, it evaluates the remaining layout across a grid of cue-ball landing positions. The resulting heatmaps favor areas that leave a playable next shot. For direct shots, coarse simulations also indicate which areas the current shot is likely to reach.

The planner ranks options using current potting ease and future position value, with adjustments for factors such as clearing difficult balls and following the intended shot sequence. It can pass a ranked option and its position target to the executor without prescribing cue parameters.

## Repository structure

| Path | Responsibility |
| --- | --- |
| [`table_vision/`](table_vision/) | Detector training, photo inference, calibration, and layout reconstruction. |
| [`cue_router/`](cue_router/) | Billiards planning and simulated shot search. |
| [`cue_router/executor/`](cue_router/executor/) | Search and score cue actions for a specified shot. |
| [`cue_router/planner/`](cue_router/planner/) | Rank shot decisions and generate cue-ball position targets. |
| [`web/`](web/) | Photo upload, layout review, and visualization of plans and simulated shots. |

## Current status

Table vision, the executor, and the planner are implemented and available through the photo-upload demo. I'm now working on a real-table setup at my school, using a fixed overhead camera to capture table states without taking and uploading a photo by hand each turn.
