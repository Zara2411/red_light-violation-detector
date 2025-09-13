# rlv_final.py
import cv2
import numpy as np
import os
import csv
from datetime import datetime
from ultralytics import YOLO

# ----------------- CONFIG -----------------
VIDEO_PATH = "videos/video1.mp4"   # set your path
MODEL_PATH = "yolov8n.pt"
OUTPUT_DIR = "violations"
LOG_CSV = os.path.join(OUTPUT_DIR, "violations.csv")

# CentroidTracker params
MAX_DISAPPEARED = 40
MAX_MATCH_DISTANCE = 80

# Default HSV values (will be tuned by the calibration UI)
HSV_SETTINGS = {
    "l1_h": 0, "u1_h": 15,
    "l2_h": 160, "u2_h": 180,
    "s_min": 40, "v_min": 40,
    "ratio_percent": 2  # fraction% of ROI required
}

ALLOWED_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- Centroid Tracker (simple) -----------------
class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 1
        self.objects = {}    # id -> centroid (x,y)
        self.bboxes = {}     # id -> bbox (x,y,w,h)
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid, bbox):
        oid = self.nextObjectID
        self.nextObjectID += 1
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        return oid

    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
            del self.bboxes[oid]
            del self.disappeared[oid]

    def update(self, rects):
        # rects: list of (x,y,w,h)
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.objects, self.bboxes

        inputCentroids = []
        for (x, y, w, h) in rects:
            cx = int(x + w / 2)
            cy = int(y + h)   # bottom-center as representative
            inputCentroids.append((cx, cy))

        if len(self.objects) == 0:
            for i, c in enumerate(inputCentroids):
                self.register(c, rects[i])
            return self.objects, self.bboxes

        objectIDs = list(self.objects.keys())
        objectCentroids = [self.objects[oid] for oid in objectIDs]

        D = np.zeros((len(objectCentroids), len(inputCentroids)), dtype="float")
        for i, oc in enumerate(objectCentroids):
            for j, ic in enumerate(inputCentroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

        # greedy matching sorted by distance
        rows = D.shape[0]
        cols = D.shape[1]
        usedRows, usedCols = set(), set()
        pairs = []

        flat = [(D[i, j], i, j) for i in range(rows) for j in range(cols)]
        flat.sort(key=lambda x: x[0])

        for dist, i, j in flat:
            if i in usedRows or j in usedCols:
                continue
            if dist > self.maxDistance:
                continue
            usedRows.add(i)
            usedCols.add(j)
            pairs.append((i, j))

        # update matched
        assignedCols = set()
        for (row, col) in pairs:
            oid = objectIDs[row]
            self.objects[oid] = inputCentroids[col]
            self.bboxes[oid] = rects[col]
            self.disappeared[oid] = 0
            assignedCols.add(col)

        # register unassigned input centroids
        for j in range(len(inputCentroids)):
            if j not in assignedCols:
                self.register(inputCentroids[j], rects[j])

        # increase disappeared for unmatched existing
        for i in range(len(objectCentroids)):
            if i not in {p[0] for p in pairs}:
                oid = objectIDs[i]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)

        return self.objects, self.bboxes

# ----------------- Interactive selectors -----------------
def select_roi(frame, window="Select TRAFFIC LIGHT ROI (tight around RED bulb)"):
    roi = cv2.selectROI(window, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window)
    x, y, w, h = roi
    return (int(x), int(y), int(w), int(h))

def select_polygon(frame, points_needed=4, window="Select LANE ROI polygon (click points)"):
    pts = []
    tmp = frame.copy()
    def cb(evt, x, y, flags, param):
        nonlocal pts, tmp
        if evt == cv2.EVENT_LBUTTONDOWN:
            pts.append((int(x), int(y)))
            cv2.circle(tmp, (x, y), 4, (0,255,0), -1)
            cv2.imshow(window, tmp)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, cb)
    cv2.imshow(window, tmp)
    print(f"Click {points_needed} points then press 'c' to confirm, 'r' to reset.")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('r'):
            pts = []
            tmp = frame.copy()
            cv2.imshow(window, tmp)
        elif k == ord('c'):
            if len(pts) >= 3:
                cv2.destroyWindow(window)
                return np.array(pts, dtype=np.int32)
            else:
                print("Need at least 3 points.")
        elif k == 27:
            cv2.destroyWindow(window)
            return None

def select_line(frame, window="Select STOP LINE (click two points)"):
    pts = []
    tmp = frame.copy()
    def cb(evt, x, y, flags, param):
        nonlocal pts, tmp
        if evt == cv2.EVENT_LBUTTONDOWN:
            pts.append((int(x), int(y)))
            cv2.circle(tmp, (x, y), 4, (0,0,255), -1)
            cv2.imshow(window, tmp)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, cb)
    cv2.imshow(window, tmp)
    print("Click 2 points for the stop-line then press 'c' to confirm.")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('r'):
            pts = []
            tmp = frame.copy()
            cv2.imshow(window, tmp)
        elif k == ord('c'):
            if len(pts) == 2:
                cv2.destroyWindow(window)
                return pts
            else:
                print("Need exactly 2 points.")
        elif k == 27:
            cv2.destroyWindow(window)
            return None

# ----------------- HSV Calibration UI -----------------
def calibrate_hsv(frame, roi):
    # roi = (x,y,w,h)
    win = "HSV CALIBRATE - adjust trackbars, press 's' to save values"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # create trackbars
    def nothing(x): pass
    cv2.createTrackbar("l1_h", win, HSV_SETTINGS["l1_h"], 180, nothing)
    cv2.createTrackbar("u1_h", win, HSV_SETTINGS["u1_h"], 180, nothing)
    cv2.createTrackbar("l2_h", win, HSV_SETTINGS["l2_h"], 180, nothing)
    cv2.createTrackbar("u2_h", win, HSV_SETTINGS["u2_h"], 180, nothing)
    cv2.createTrackbar("s_min", win, HSV_SETTINGS["s_min"], 255, nothing)
    cv2.createTrackbar("v_min", win, HSV_SETTINGS["v_min"], 255, nothing)
    cv2.createTrackbar("ratio%", win, HSV_SETTINGS["ratio_percent"], 100, nothing)

    while True:
        x, y, w, h = roi
        roi_img = frame[y:y+h, x:x+w].copy()
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        l1 = cv2.getTrackbarPos("l1_h", win)
        u1 = cv2.getTrackbarPos("u1_h", win)
        l2 = cv2.getTrackbarPos("l2_h", win)
        u2 = cv2.getTrackbarPos("u2_h", win)
        s_min = cv2.getTrackbarPos("s_min", win)
        v_min = cv2.getTrackbarPos("v_min", win)
        ratio_p = cv2.getTrackbarPos("ratio%", win)

        lower1 = np.array([l1, s_min, v_min])
        upper1 = np.array([u1, 255, 255])
        lower2 = np.array([l2, s_min, v_min])
        upper2 = np.array([u2, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = mask1 | mask2

        # annotate
        try:
            ratio = np.count_nonzero(mask) / float(w*h)
        except ZeroDivisionError:
            ratio = 0.0

        display = roi_img.copy()
        cv2.putText(display, f"ratio={ratio:.3f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("ROI (calibrate)", display)
        cv2.imshow("HSV mask (calibrate)", mask)

        k = cv2.waitKey(50) & 0xFF
        if k == ord('s'):   # save and exit
            HSV_SETTINGS.update({
                "l1_h": l1, "u1_h": u1,
                "l2_h": l2, "u2_h": u2,
                "s_min": s_min, "v_min": v_min,
                "ratio_percent": ratio_p
            })
            cv2.destroyWindow("ROI (calibrate)")
            cv2.destroyWindow("HSV mask (calibrate)")
            cv2.destroyWindow(win)
            print("Saved HSV settings:", HSV_SETTINGS)
            return
        elif k == 27:
            # ESC to cancel (keep defaults)
            cv2.destroyWindow("ROI (calibrate)")
            cv2.destroyWindow("HSV mask (calibrate)")
            cv2.destroyWindow(win)
            print("Calibration cancelled; using defaults:", HSV_SETTINGS)
            return

# ----------------- Utility functions -----------------
def red_on_from_frame(frame, roi):
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return False, 0.0
    roi_img = frame[y:y+h, x:x+w]
    if roi_img.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([HSV_SETTINGS["l1_h"], HSV_SETTINGS["s_min"], HSV_SETTINGS["v_min"]])
    upper1 = np.array([HSV_SETTINGS["u1_h"], 255, 255])
    lower2 = np.array([HSV_SETTINGS["l2_h"], HSV_SETTINGS["s_min"], HSV_SETTINGS["v_min"]])
    upper2 = np.array([HSV_SETTINGS["u2_h"], 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    red_pixels = int(np.count_nonzero(mask))
    ratio = red_pixels / float(max(1, w*h))
    # debug windows
    cv2.imshow("Traffic Light ROI", roi_img)
    cv2.imshow("Traffic Light Mask", mask)
    threshold = HSV_SETTINGS["ratio_percent"] / 100.0
    return ratio >= threshold, red_pixels

def point_in_polygon(point, poly):
    return cv2.pointPolygonTest(poly, point, False) >= 0

def line_y_at_x(line_pts, x):
    (xA, yA), (xB, yB) = line_pts
    m = (yB - yA) / (xB - xA + 1e-6)
    return int(m * x + (yA - m * xA))

# ----------------- Logging helpers -----------------
def ensure_log_csv():
    header = ["timestamp","object_id","frame_idx","image_path","bbox_x1","bbox_y1","bbox_x2","bbox_y2"]
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def append_log(entry):
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(entry)

# ----------------- Main processing -----------------
def main():
    print("Loading YOLO model (this may download weights first time)...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    ok, first = cap.read()
    if not ok:
        print("❌ Cannot open video:", VIDEO_PATH)
        return

    # 1) select traffic light ROI
    print("\nSTEP 1: Select traffic-light ROI (tight around RED bulb).")
    tl_roi = select_roi(first)

    # 2) calibrate HSV on that ROI
    print("\nSTEP 2: Calibrate HSV for red detection. Use 's' to save, 'ESC' to skip/cancel.")
    calibrate_hsv(first, tl_roi)

    # 3) select lane ROI polygon
    print("\nSTEP 3: Select lane ROI polygon (click at least 3 points). Press 'c' when done.")
    lane_roi = select_polygon(first)

    # 4) select stop line (2 points)
    print("\nSTEP 4: Select stop line (2 clicks). Press 'c' when done.")
    stop_line = select_line(first)
    if stop_line is None:
        print("Stop-line not selected. Exiting.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ct = CentroidTracker(maxDisappeared=MAX_DISAPPEARED, maxDistance=MAX_MATCH_DISTANCE)
    vehicle_states = {}  # id -> {'was_behind': bool, 'violation': bool}
    saved_violations = set()
    ensure_log_csv()
    frame_idx = 0
    prev_red = False

    print("\nProcessing... press 'q' to stop early. Watch the debug windows (ROI + Mask).")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        red_on, red_pixels = red_on_from_frame(frame, tl_roi)

        # run YOLO detection (single-frame)
        dets = model(frame, verbose=False)[0]
        rects = []
        for b in dets.boxes:
            cls = int(b.cls[0])
            if cls in ALLOWED_CLASSES:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                rects.append((x1, y1, x2 - x1, y2 - y1))

        # update tracker
        objects, bboxes = ct.update(rects)

        # if red just turned on, mark which tracked vehicles were behind
        if red_on and not prev_red:
            for oid, bbox in bboxes.items():
                x_b, y_b, w_b, h_b = bbox
                bottom_center = (int(x_b + w_b/2), int(y_b + h_b))
                inside = point_in_polygon(bottom_center, lane_roi) and point_in_polygon(bottom_center, np.array(stop_line))
                # if not inside stop polygon, it is behind (we only used lane_roi here)
                was_behind = not point_in_polygon(bottom_center, np.array(stop_line))
                vehicle_states[oid] = {'was_behind': was_behind, 'violation': False}
        # If green resumed we clear 'was_behind' flags (fresh marking on next red)
        if not red_on and prev_red:
            for oid in list(vehicle_states.keys()):
                vehicle_states[oid]['was_behind'] = False
                vehicle_states[oid]['violation'] = False

        # examine each tracked vehicle
        detections_in_roi = 0
        for oid, centroid in objects.items():
            bbox = bboxes.get(oid)
            if bbox is None:
                continue
            x_b, y_b, w_b, h_b = bbox
            bottom_center = (int(x_b + w_b/2), int(y_b + h_b))

            # only consider vehicles inside lane ROI
            if not point_in_polygon(bottom_center, lane_roi):
                continue
            detections_in_roi += 1

            # draw box & id
            cv2.rectangle(frame, (x_b, y_b), (x_b + w_b, y_b + h_b), (0,200,0), 2)
            cv2.putText(frame, f"ID {oid}", (x_b, y_b-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.circle(frame, bottom_center, 3, (255,0,0), -1)

            # ensure state exists
            if oid not in vehicle_states:
                vehicle_states[oid] = {'was_behind': False, 'violation': False}

            # violation logic: only while red is ON
            if red_on and vehicle_states[oid].get('was_behind', False) and not vehicle_states[oid]['violation']:
                # inside stop polygon?
                inside_stop = point_in_polygon(bottom_center, np.array(stop_line))
                # alternative numeric check: compare bottom y against line y at this cx
                ly = line_y_at_x(stop_line, bottom_center[0])
                beyond_line = bottom_center[1] < ly  # smaller y = higher on frame = past line (camera dependent)
                # use polygon test (inside_stop) primarily
                if inside_stop or beyond_line:
                    # log once per tracked object
                    vehicle_states[oid]['violation'] = True
                    if oid not in saved_violations:
                        saved_violations.add(oid)
                        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        fname = os.path.join(OUTPUT_DIR, f"violation_obj{oid}_frame{frame_idx}_{now}.jpg")
                        # annotate a copy for saving
                        save_frame = frame.copy()
                        cv2.putText(save_frame, f"VIOLATION ID {oid}", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                        cv2.imwrite(fname, save_frame)
                        # log csv: timestamp,object_id,frame_idx,image,bbox
                        append_log([now, oid, frame_idx, fname, x_b, y_b, x_b+w_b, y_b+h_b])
                        print(f"[LOG] Violation -> obj {oid} frame {frame_idx} saved {fname}")

        # DEBUG overlay
        cv2.polylines(frame, [lane_roi], True, (255,255,0), 2)
        cv2.line(frame, tuple(stop_line[0]), tuple(stop_line[1]), (0,0,255), 2)
        cv2.putText(frame, f"Red: {red_on} ({red_pixels})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if red_on else (0,200,0), 2)
        cv2.putText(frame, f"Frame: {frame_idx} Dets_inROI: {detections_in_roi} Tracked:{len(objects)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("RLV - Main", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

        prev_red = red_on

    cap.release()
    cv2.destroyAllWindows()
    print("Processing finished. Logs saved at:", LOG_CSV)

if __name__ == "__main__":
    main()
