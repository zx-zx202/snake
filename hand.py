import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cvzone
from cvzone.HandTrackingModule import HandDetector


class ChineseTextRenderer:
    def __init__(self):
        # 尝试加载中文字体
        self.font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
            "C:/Windows/Fonts/simsun.ttc",  # Windows 宋体
            "/System/Library/Fonts/PingFang.ttc",  # Mac 苹方
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
            "simhei.ttf"  # 当前目录
        ]
        self.font = None

        for font_path in self.font_paths:
            try:
                self.font = ImageFont.truetype(font_path, 30)
                print(f"成功加载字体: {font_path}")
                break
            except:
                continue

        if self.font is None:
            print("警告：无法加载中文字体，将使用默认字体")

    def put_chinese_text(self, img, text, position, font_size=30, color=(255, 255, 255)):
        """在图像上绘制中文文本"""
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 设置字体
        try:
            if self.font:
                font = self.font.font_variant(size=font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # 绘制文本
        draw.text(position, text, font=font, fill=color)

        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class GestureRecognizer:
    def __init__(self):
        self.detector = HandDetector(detectionCon=0.7, maxHands=2)
        self.text_renderer = ChineseTextRenderer()

        # 手势定义（中文）
        self.gestures = {
            "open_hand": "张开手掌",
            "fist": "握拳",
            "pointing": "食指指向",
            "victory": "胜利手势",
            "ok": "OK手势",
            "thumbs_up": "点赞",
            "thumbs_down": "点踩",
            "call_me": "打电话",
            "unknown": "未知手势"
        }

    def calculate_distance(self, point1, point2):
        """计算两点之间的欧几里得距离"""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def recognize_gesture(self, lmList):
        """识别手势"""
        if not lmList or len(lmList) < 21:
            return "unknown"

        # 获取关键点坐标
        thumb_tip = lmList[4]
        index_tip = lmList[8]
        middle_tip = lmList[12]
        ring_tip = lmList[16]
        pinky_tip = lmList[20]

        thumb_ip = lmList[3]
        index_pip = lmList[6]
        middle_pip = lmList[10]
        ring_pip = lmList[14]
        pinky_pip = lmList[18]

        # 计算手指是否伸直
        finger_distances = [
            self.calculate_distance(thumb_tip, thumb_ip),
            self.calculate_distance(index_tip, index_pip),
            self.calculate_distance(middle_tip, middle_pip),
            self.calculate_distance(ring_tip, ring_pip),
            self.calculate_distance(pinky_tip, pinky_pip)
        ]

        finger_extended_threshold = 40
        fingers = [1 if dist > finger_extended_threshold else 0 for dist in finger_distances]

        # 手势识别逻辑
        if fingers == [0, 1, 0, 0, 0]:
            return "pointing"
        elif fingers == [1, 1, 0, 0, 0]:
            return "pointing"
        elif fingers == [0, 1, 1, 0, 0]:
            return "victory"
        elif fingers == [1, 1, 1, 1, 1]:
            return "open_hand"
        elif fingers == [0, 0, 0, 0, 0]:
            return "fist"
        elif fingers == [1, 0, 0, 0, 1]:
            return "call_me"
        else:
            thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
            if thumb_index_dist < 35 and fingers[2:] == [0, 0, 0]:
                return "ok"
            if fingers[0] == 1 and fingers[1:] == [0, 0, 0, 0]:
                return "thumbs_up"

        return "unknown"

    def draw_hand_info(self, img, hands, gesture):
        """在图像上绘制手部信息和手势（中文）"""
        if hands:
            for hand in hands:
                bbox = hand["bbox"]
                x, y, w, h = bbox

                # 绘制边界框
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 使用中文显示手势信息
                hand_type = "左手" if hand["type"] == "Left" else "右手"
                gesture_chinese = self.gestures.get(gesture, "未知手势")
                info_text = f"{hand_type}: {gesture_chinese}"

                # 使用中文渲染器绘制文本
                img = self.text_renderer.put_chinese_text(
                    img, info_text, (x, y - 40), font_size=20, color=(255, 255, 0)
                )

                # 绘制关键点连接线
                self.draw_hand_connections(img, hand)

    def draw_hand_connections(self, img, hand):
        """绘制手部关键点连接线"""
        lmList = hand["lmList"]

        connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20],
            [5, 9], [9, 13], [13, 17]
        ]

        for connection in connections:
            if connection[0] < len(lmList) and connection[1] < len(lmList):
                start_point = tuple(lmList[connection[0]][0:2])
                end_point = tuple(lmList[connection[1]][0:2])
                cv2.line(img, start_point, end_point, (255, 0, 255), 2)

        for lm in lmList:
            x, y = lm[0:2]
            cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    gesture_recognizer = GestureRecognizer()

    print("手势识别演示 - 中文版")
    print("支持的手势：")
    for eng, chn in gesture_recognizer.gestures.items():
        print(f"  {eng}: {chn}")
    print("按 'Q' 键退出")

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        hands, img = gesture_recognizer.detector.findHands(img, flipType=False, draw=False)

        current_gesture = "unknown"

        if hands:
            lmList = hands[0]['lmList']
            current_gesture = gesture_recognizer.recognize_gesture(lmList)
            gesture_recognizer.draw_hand_info(img, hands, current_gesture)

        # 显示当前识别到的手势（中文）
        gesture_chinese = gesture_recognizer.gestures.get(current_gesture, "未知手势")
        img = gesture_recognizer.text_renderer.put_chinese_text(
            img, f'识别手势: {gesture_chinese}', (50, 50),
            font_size=40, color=(255, 255, 255)
        )

        # 显示操作提示（中文）
        img = gesture_recognizer.text_renderer.put_chinese_text(
            img, "按 'Q' 键退出", (50, 100),
            font_size=25, color=(0, 255, 255)
        )

        cv2.imshow("MediaPipe 手势识别 - 中文版", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()