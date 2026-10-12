import os
"""
TASK 4 V2 - Fixed-ID Grounding DINO tracking

Grounding DINO Tiny -> constrained fixed two-track Kalman tracker.
No new IDs are created after initialization. This prevents the V1
SORT failure where temporary detector failures created IDs 3..13.
"""
from pathlib import Path
import csv, time
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

PROJECT_ROOT = Path(Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent)))
VIDEO_PATH = PROJECT_ROOT / "Raw and Boiled Egg Spinning - Cool Science Experiment ｜ Educational Videos by Mocomi [Y4JyV94y2_A].webm"
OUTPUT_DIR = PROJECT_ROOT / "task4_v2_fixed_ids"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "task4_tracking_v2_test.mp4"
METRICS_CSV = OUTPUT_DIR / "task4_tracking_v2_metrics.csv"

TEST_ONLY = True
START_TIME, END_TIME = 18.0, 34.0
DETECTION_INTERVAL = 6
DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "egg."
BOX_THRESHOLD, TEXT_THRESHOLD = 0.25, 0.20
DETECTOR_WIDTH = 960
MIN_BOX_AREA = 2500
MAX_CENTER_DISTANCE = 220.0
MIN_SIZE_RATIO, MAX_SIZE_RATIO = 0.45, 2.20
MIN_IOU_FOR_DIRECT_MATCH = 0.08
MIN_ASSIGNMENT_SCORE = 0.32
MAX_PREDICTION_AGE = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Boxes at 18 s from the controlled test window.
INITIAL_TRACKS = [
    (1, np.array([75., 522., 271., 718.], dtype=np.float32)),
    (2, np.array([350., 547., 560., 757.], dtype=np.float32)),
]

def xyxy_to_xywh(b):
    x1,y1,x2,y2=b; return np.array([(x1+x2)/2,(y1+y2)/2,max(1,x2-x1),max(1,y2-y1)],np.float32)

def xywh_to_xyxy(b):
    cx,cy,w,h=b; return np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2],np.float32)

def center(b): return np.array([(b[0]+b[2])/2,(b[1]+b[3])/2],np.float32)
def size(b): return np.array([max(1,b[2]-b[0]),max(1,b[3]-b[1])],np.float32)
def area(b): return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); u=area(a)+area(b)-inter
    return float(inter/u) if u>0 else 0.0

class KalmanBox:
    def __init__(self, box):
        self.kf=cv2.KalmanFilter(8,4)
        self.kf.transitionMatrix=np.eye(8,dtype=np.float32)
        for i in range(4): self.kf.transitionMatrix[i,i+4]=1
        self.kf.measurementMatrix=np.zeros((4,8),np.float32)
        for i in range(4): self.kf.measurementMatrix[i,i]=1
        self.kf.processNoiseCov=np.eye(8,dtype=np.float32)*1e-2
        self.kf.measurementNoiseCov=np.eye(4,dtype=np.float32)*0.1
        self.kf.errorCovPost=np.eye(8,dtype=np.float32)
        self.kf.statePost=np.zeros((8,1),np.float32)
        self.kf.statePost[:4]=xyxy_to_xywh(box).reshape(4,1)
        self.last_box=box.copy()
    def predict(self):
        self.last_box=xywh_to_xyxy(self.kf.predict().reshape(-1)[:4]); return self.last_box.copy()
    def update(self,box):
        self.kf.correct(xyxy_to_xywh(box).reshape(4,1)); self.last_box=box.copy()

class FixedTrack:
    def __init__(self,track_id,box):
        self.track_id=track_id; self.kalman=KalmanBox(box); self.current_box=box.copy()
        self.expected_size=size(box); self.state="INITIALIZED"; self.missed=0
        self.matched=0; self.predicted=0; self.lost=0; self.last_score=0.0
    def predict(self):
        self.current_box=self.kalman.predict(); self.missed+=1
        if self.missed>MAX_PREDICTION_AGE: self.state="LOST"; self.lost+=1
        else: self.state="PREDICTED"; self.predicted+=1
    def update(self,box,score):
        self.kalman.update(box); self.current_box=box.copy(); self.expected_size=.15*size(box)+.85*self.expected_size
        self.missed=0; self.matched+=1; self.last_score=float(score); self.state="TRACKING"

def load_model():
    print("="*70); print("LOADING GROUNDING DINO TINY"); print("="*70)
    print(f"Model: {DINO_MODEL_ID}\nDevice: {DEVICE}")
    p=AutoProcessor.from_pretrained(DINO_MODEL_ID)
    m=AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(DEVICE); m.eval()
    return p,m

@torch.no_grad()
def detect(frame,processor,model):
    oh,ow=frame.shape[:2]; scale=min(1.0,DETECTOR_WIDTH/float(ow))
    if scale<1:
        dw,dh=int(round(ow*scale)),int(round(oh*scale)); img=cv2.resize(frame,(dw,dh),interpolation=cv2.INTER_AREA)
    else: dw,dh=ow,oh; img=frame
    img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    inp=processor(images=img,text=TEXT_PROMPT,return_tensors="pt")
    inp={k:(v.to(DEVICE) if hasattr(v,"to") else v) for k,v in inp.items()}
    out=model(**inp)
    try: r=processor.post_process_grounded_object_detection(out,inp["input_ids"],threshold=BOX_THRESHOLD,text_threshold=TEXT_THRESHOLD,target_sizes=[(dh,dw)])[0]
    except TypeError: r=processor.post_process_grounded_object_detection(out,inp["input_ids"],box_threshold=BOX_THRESHOLD,text_threshold=TEXT_THRESHOLD,target_sizes=[(dh,dw)])[0]
    boxes=r["boxes"].detach().cpu().numpy(); scores=r["scores"].detach().cpu().numpy()
    if scale<1: boxes/=scale
    ds=[]
    for b,s in zip(boxes,scores):
        b=np.array([max(0,min(float(b[0]),ow-1)),max(0,min(float(b[1]),oh-1)),max(0,min(float(b[2]),ow-1)),max(0,min(float(b[3]),oh-1))],np.float32)
        if area(b)>=MIN_BOX_AREA: ds.append({"box":b,"score":float(s)})
    ds.sort(key=lambda x:x["score"],reverse=True); return ds

def candidate_score(track,c):
    pb,cb=track.current_box,c["box"]; d=float(np.linalg.norm(center(cb)-center(pb)))
    if d>MAX_CENTER_DISTANCE: return -1
    ov=iou(pb,cb); ew,eh=track.expected_size; cw,ch=size(cb)
    wr,hr=cw/max(1,ew),ch/max(1,eh)
    if not(MIN_SIZE_RATIO<=wr<=MAX_SIZE_RATIO and MIN_SIZE_RATIO<=hr<=MAX_SIZE_RATIO): return -1
    ssize=float(np.exp(-(abs(np.log(max(wr,1e-6)))+abs(np.log(max(hr,1e-6))))))
    ds=max(0,1-d/MAX_CENTER_DISTANCE); iscore=min(1,ov/.5); conf=c["score"]
    score=.35*iscore+.30*ds+.20*ssize+.15*conf
    if ov<MIN_IOU_FOR_DIRECT_MATCH and score<.48: return -1
    return score if score>=MIN_ASSIGNMENT_SCORE else -1

def associate(tracks,detections):
    if not detections: return
    scores=np.full((len(tracks),len(detections)),-1.,np.float32)
    for i,t in enumerate(tracks):
        for j,d in enumerate(detections): scores[i,j]=candidate_score(t,d)
    rows,cols=linear_sum_assignment(1-np.maximum(scores,0))
    used=set()
    for r,c in zip(rows,cols):
        if scores[r,c]>=MIN_ASSIGNMENT_SCORE and c not in used:
            tracks[r].update(detections[c]["box"],detections[c]["score"]); used.add(c)

def draw(frame,t):
    x1,y1,x2,y2=[int(round(v)) for v in t.current_box]; h,w=frame.shape[:2]
    x1,y1=max(0,min(x1,w-1)),max(0,min(y1,h-1)); x2,y2=max(0,min(x2,w-1)),max(0,min(y2,h-1))
    color=(0,220,0) if t.state=="TRACKING" else ((0,220,255) if t.state=="PREDICTED" else (0,0,255))
    state={"TRACKING":"MATCHED","PREDICTED":"PREDICTED","LOST":"LOST"}.get(t.state,t.state)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
    cv2.putText(frame,f"Egg ID {t.track_id} | {state}",(x1,max(30,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,.72,color,2,cv2.LINE_AA)

def main():
    if not VIDEO_PATH.exists(): raise FileNotFoundError(VIDEO_PATH)
    processor,model=load_model(); cap=cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened(): raise RuntimeError("Could not open video")
    fps=cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); duration=total/fps
    start=START_TIME if TEST_ONLY else 0.; end=min(END_TIME,duration) if TEST_ONLY else duration
    print(f"Resolution: {w} x {h}\nFPS: {fps:.2f}\nDuration: {duration:.2f}s\nProcessing: {start:.2f}s -> {end:.2f}s")
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(round(start*fps)))
    writer=cv2.VideoWriter(str(OUTPUT_VIDEO),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    tracks=[FixedTrack(i,b) for i,b in INITIAL_TRACKS]
    frame=int(round(start*fps)); processed=0; det_runs=0; total_det=0; accepted=0; t0=time.perf_counter()
    while frame<=int(round(end*fps)):
        ok,img=cap.read()
        if not ok: break
        now=frame/fps
        for t in tracks: t.predict()
        ds=[]
        if processed%DETECTION_INTERVAL==0:
            ds=detect(img,processor,model); det_runs+=1; total_det+=len(ds); before=[t.matched for t in tracks]; associate(tracks,ds); accepted+=sum(t.matched-b for t,b in zip(tracks,before))
        for t in tracks: draw(img,t)
        cv2.putText(img,f"Time: {now:.2f}s",(25,40),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img,"Grounding DINO + Fixed-ID Tracker V2",(25,75),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2,cv2.LINE_AA)
        writer.write(img); frame+=1; processed+=1
        if processed%max(1,int(fps*2))==0:
            print(f"time={now:.2f}s | frames={processed} | det_runs={det_runs} | detections={len(ds)} | ID1={tracks[0].state} | ID2={tracks[1].state} | runtime={time.perf_counter()-t0:.1f}s")
    cap.release(); writer.release(); elapsed=time.perf_counter()-t0
    with open(METRICS_CSV,"w",newline="",encoding="utf-8") as f:
        cw=csv.writer(f); cw.writerow(["track_id","matched_frames","predicted_frames","lost_frames","processed_frames"])
        for t in tracks: cw.writerow([t.track_id,t.matched,t.predicted,t.lost,processed])
    print("="*70); print("V2 TRACKING COMPLETE"); print(f"Processed frames: {processed}"); print(f"Detector runs: {det_runs}"); print(f"Total detections: {total_det}"); print(f"Accepted updates: {accepted}"); print(f"Runtime: {elapsed:.2f}s"); print(f"Processing FPS: {processed/elapsed:.2f}");
    for t in tracks: print(f"Egg ID {t.track_id}: matched={t.matched}, predicted={t.predicted}, lost={t.lost}")
    print(f"Video: {OUTPUT_VIDEO}\nMetrics: {METRICS_CSV}")

if __name__=="__main__": main()
