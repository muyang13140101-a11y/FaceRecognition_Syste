import cv2
from core.detector import FaceDetector

def main():
    # 1. 实例化我们刚刚写好的检测器（此时 MTCNN 模型加载进内存）
    print("正在加载模型，请稍候...")
    detector = FaceDetector()
    
    # 2. 打开电脑的默认摄像头 (0 代表第一个摄像头)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return

    print("摄像头已启动，按 'q' 键退出。")

    # 3. 进入死循环，不断读取摄像头的每一帧画面
    while True:
        # ret 是个布尔值，代表是否成功读到画面；frame 就是当前的图片矩阵
        ret, frame = cap.read()
        if not ret:
            break

        # 4. 调用我们写的模块，把画面传进去抠脸
        # 注意：这里我们只要人脸的坐标框来画图，暂时不需要抠出来的像素
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detector.detect_faces(rgb_image)

        # 5. 把检测到的结果画在画面上
        for result in results:
            if result['confidence'] > 0.95:
                x, y, w, h = result['box']
                # 用绿色 (0, 255, 0) 画一个矩形框，线条粗细为 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 可选：把 5 个关键点也画出来（红色小圆点）
                keypoints = result['keypoints']
                cv2.circle(frame, keypoints['left_eye'], 2, (0, 0, 255), 2)
                cv2.circle(frame, keypoints['right_eye'], 2, (0, 0, 255), 2)
                cv2.circle(frame, keypoints['nose'], 2, (0, 0, 255), 2)
                cv2.circle(frame, keypoints['mouth_left'], 2, (0, 0, 255), 2)
                cv2.circle(frame, keypoints['mouth_right'], 2, (0, 0, 255), 2)

        # 6. 显示画面
        cv2.imshow("MTCNN CPU Test", frame)

        # 7. 监听键盘事件，按下 'q' 键就打破循环退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()