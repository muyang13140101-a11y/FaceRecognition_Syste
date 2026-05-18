import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import FaceRecognitionApp

def main():
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()