import cv2
import mediapipe as mp
import numpy as np

class PoseAnalyzer:
    def __init__(self, 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5,
                 wrist_nose_threshold=200,
                 thumb_palm_threshold=50,
                 wrist_rotation_threshold=5,
                 elbow_angle_threshold=90):
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Thresholds
        self.WRIST_NOSE_THRESHOLD = wrist_nose_threshold
        self.THUMB_PALM_THRESHOLD = thumb_palm_threshold
        self.WRIST_ROTATION_THRESHOLD = wrist_rotation_threshold
        self.ELBOW_ANGLE_THRESHOLD = elbow_angle_threshold
        
        self.prev_wrist_angle_right = None
        self.prev_wrist_angle_left = None
    
    @staticmethod
    def calculate_angle(a, b, c):
        """محاسبه زاویه بین سه نقطه"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(np.degrees(radians))
        return angle if angle <= 180 else 360 - angle
    
    def calculate_wrist_rotation(self, elbow, wrist, prev_angle):
        """محاسبه چرخش مچ دست"""
        vec = np.array(wrist) - np.array(elbow)
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        
        if prev_angle is None:
            return angle, 0
        
        rotation = abs(angle - prev_angle)
        return angle, rotation
    
    def analyze_hand(self, landmarks, side='right', frame_width=None, frame_height=None):
        """آنالیز وضعیت یک دست"""
        results = {
            'status': [],
            'landmarks': {},
            'angles': {},
            'distances': {}
        }
        
        if frame_width is None or frame_height is None:
            w, h = 1, 1  # مقادیر پیش‌فرض
        else:
            w, h = frame_width, frame_height
        
        shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER if side == 'left' else self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW if side == 'left' else self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST if side == 'left' else self.mp_pose.PoseLandmark.RIGHT_WRIST]
        thumb = landmarks[self.mp_pose.PoseLandmark.LEFT_THUMB if side == 'left' else self.mp_pose.PoseLandmark.RIGHT_THUMB]
        pinky = landmarks[self.mp_pose.PoseLandmark.LEFT_PINKY if side == 'left' else self.mp_pose.PoseLandmark.RIGHT_PINKY]
        
        key_points = {
            'shoulder': [shoulder.x * w, shoulder.y * h],
            'elbow': [elbow.x * w, elbow.y * h],
            'wrist': [wrist.x * w, wrist.y * h],
            'thumb': [thumb.x * w, thumb.y * h],
            'pinky': [pinky.x * w, pinky.y * h]
        }
        
        results['landmarks'] = key_points
        
        # محاسبه زاویه آرنج
        elbow_angle = self.calculate_angle(
            key_points['shoulder'], 
            key_points['elbow'], 
            key_points['wrist']
        )
        results['angles']['elbow'] = elbow_angle
        
        if elbow_angle < self.ELBOW_ANGLE_THRESHOLD:
            results['status'].append(f"{side.capitalize()} elbow bent")
        
        # محاسبه چرخش مچ دست
        prev_angle = self.prev_wrist_angle_left if side == 'left' else self.prev_wrist_angle_right
        wrist_angle, wrist_rotation = self.calculate_wrist_rotation(
            key_points['elbow'], 
            key_points['wrist'],
            prev_angle
        )
        
        if side == 'left':
            self.prev_wrist_angle_left = wrist_angle
        else:
            self.prev_wrist_angle_right = wrist_angle
            
        results['angles']['wrist_rotation'] = wrist_rotation
        
        if wrist_rotation > self.WRIST_ROTATION_THRESHOLD:
            results['status'].append(f"{side.capitalize()} wrist rotated")
        
        # محاسبه فاصله شست تا کف دست
        thumb_palm_dist = np.linalg.norm(
            np.array(key_points['thumb']) - np.array(key_points['pinky'])
        )
        results['distances']['thumb_palm'] = thumb_palm_dist
        
        if thumb_palm_dist < self.THUMB_PALM_THRESHOLD:
            results['status'].append(f"{side.capitalize()} thumb near palm")
        
        return results
    
    def analyze_frame(self, frame):
        """آنالیز وضعیت بدن در فریم فعلی"""
        results = {
            'right_hand': {},
            'left_hand': {},
            'nose': None,
            'status': []
        }
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(image)
        
        if not pose_results.pose_landmarks:
            return results
        
        h, w = frame.shape[:2]
        landmarks = pose_results.pose_landmarks.landmark
        
        # نقطه بینی (مشترک برای هر دو دست)
        nose = [landmarks[self.mp_pose.PoseLandmark.NOSE].x * w,
               landmarks[self.mp_pose.PoseLandmark.NOSE].y * h]
        results['nose'] = nose
        
        # آنالیز دست راست
        right_hand = self.analyze_hand(landmarks, 'right', w, h)
        results['right_hand'] = right_hand
        
        # آنالیز دست چپ
        left_hand = self.analyze_hand(landmarks, 'left', w, h)
        results['left_hand'] = left_hand
        
        # محاسبه فاصله مچ‌ها تا بینی
        for side in ['right', 'left']:
            hand = results[f'{side}_hand']
            if 'wrist' in hand['landmarks']:
                wrist_nose_dist = np.linalg.norm(
                    np.array(hand['landmarks']['wrist']) - np.array(nose)
                )
                hand['distances']['wrist_nose'] = wrist_nose_dist
                
                if wrist_nose_dist < self.WRIST_NOSE_THRESHOLD:
                    results['status'].append(f"{side.capitalize()} wrist near face")
        
        # جمع‌بندی وضعیت‌ها
        results['status'].extend(right_hand['status'])
        results['status'].extend(left_hand['status'])
        
        return results


class PoseVisualizer:
    @staticmethod
    def draw_analysis_results(frame, analysis_results):
        """رسم نتایج آنالیز روی فریم"""
        colors = {
            'right': (0, 255, 0),  # سبز برای دست راست
            'left': (0, 0, 255)    # قرمز برای دست چپ
        }
        
        # رسم خطوط و نقاط برای هر دست
        for side in ['right', 'left']:
            hand = analysis_results.get(f'{side}_hand', {})
            landmarks = hand.get('landmarks', {})
            
            if not landmarks:
                continue
            
            color = colors[side]
            
            # رسم خطوط بین نقاط
            connections = [
                ('shoulder', 'elbow'),
                ('elbow', 'wrist'),
                ('wrist', 'thumb'),
                ('wrist', 'pinky')
            ]
            
            for start, end in connections:
                if start in landmarks and end in landmarks:
                    cv2.line(frame, 
                            (int(landmarks[start][0]), int(landmarks[start][1])),
                            (int(landmarks[end][0]), int(landmarks[end][1])),
                            color, 2)
            
            # رسم نقاط کلیدی
            for point in landmarks.values():
                cv2.circle(frame, (int(point[0]), int(point[1])), 8, color, -1)
        
        # نمایش وضعیت
        status_text = ", ".join(analysis_results.get('status', []))
        cv2.putText(frame, status_text, (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame


if __name__ == "__main__":
    # تنظیمات برای دوربین مداربسته
    pose_analyzer = PoseAnalyzer(
        wrist_nose_threshold=250,  
        thumb_palm_threshold=7,
        wrist_rotation_threshold=7,
        elbow_angle_threshold=100
    )
    
    visualizer = PoseVisualizer()
    
    # برای دوربین مداربسته می‌توانید از آدرس IP دوربین استفاده کنید
    cap = cv2.VideoCapture(0)  # یا "rtsp://username:password@ip_address:port"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # آنالیز فریم
        analysis = pose_analyzer.analyze_frame(frame)
        
        # نمایش نتایج
        frame = visualizer.draw_analysis_results(frame, analysis)
        
        cv2.imshow("Pose Analysis (Right: Green, Left: Red)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()