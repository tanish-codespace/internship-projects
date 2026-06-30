import cv2, csv, os, numpy as np, mediapipe as mp

WORDS            = ["Hello", "Yes", "No", "ILoveYou", "Water"]
SAMPLES_PER_WORD = 400
OUT_CSV          = "landmark_data.csv"

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
HINTS = {"Hello":"Flat hand at temple","Yes":"Closed FIST",
         "No":"Index+middle tap THUMB","ILoveYou":"Thumb+index+pinky out",
         "Water":"W-hand (3 fingers) at CHIN"}

def normalize(lms):
    pts = np.array(lms, dtype=np.float32)
    pts = pts - pts[0]
    m = np.max(np.linalg.norm(pts, axis=1))
    if m > 0: pts = pts / m
    return pts.flatten()

def put(f, t, y, c=(255,255,255), s=0.8, th=2):
    cv2.putText(f, t, (15,y), cv2.FONT_HERSHEY_SIMPLEX, s, (0,0,0), th+3, cv2.LINE_AA)
    cv2.putText(f, t, (15,y), cv2.FONT_HERSHEY_SIMPLEX, s, c, th, cv2.LINE_AA)

def main():
    new = not os.path.exists(OUT_CSV)
    f = open(OUT_CSV, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow([f"{a}{i}" for i in range(21) for a in ("x","y")] + ["label"])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: webcam not accessible."); return
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

    for word in WORDS:
        collecting, count = False, 0
        while count < SAMPLES_PER_WORD:
            ok, frame = cap.read()
            if not ok: continue
            frame = cv2.flip(frame, 1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            found = False
            if res.multi_hand_landmarks:
                found = True
                hl = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                if collecting:
                    coords = [(lm.x, lm.y) for lm in hl.landmark]
                    w.writerow(list(normalize(coords)) + [word]); count += 1
            put(frame, f"SIGN: {word}", 35, (0,255,255), 1.0)
            put(frame, HINTS.get(word,""), 70, (255,255,255), 0.6, 1)
            if collecting:
                put(frame, f"Collecting {count}/{SAMPLES_PER_WORD}", 105, (0,255,0), 0.8)
                put(frame, "Move hand around a little", 140, (200,200,200), 0.55, 1)
            else:
                put(frame, "SPACE=start  n=skip  q=quit", 105, (0,200,255), 0.7)
            if not found:
                put(frame, "No hand detected", 175, (0,0,255), 0.7)
            cv2.imshow("Capture", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '): collecting = True
            elif k == ord('n'): break
            elif k == ord('q'):
                cap.release(); cv2.destroyAllWindows(); f.close(); print("Quit."); return
        print(f"Done: {word} ({count})")

    cap.release(); cv2.destroyAllWindows(); hands.close(); f.close()
    print(f"\nSaved -> {OUT_CSV}\nNext: python train_landmarks.py")

if __name__ == "__main__":
    main()