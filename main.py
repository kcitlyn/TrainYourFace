from display import display_manager
import cv2

def main():
    display= display_manager.DisplayManager()
    display.training_display()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        exit()
    