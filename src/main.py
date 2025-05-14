import sys
from PyQt6.QtWidgets import QApplication
from ui import DrowsinessDetectionUI
from config import APP_CONFIG

def main():
    app = QApplication(sys.argv)
    window = DrowsinessDetectionUI(APP_CONFIG)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()