from display import display_manager
import cv2
from display import utils

def main():
    utils.make_json_if_unavailable("properties")
    utils.make_json_if_unavailable("descriptors")
    while True:
        print("1 for training; 2 for identification")
        instruction_mode= utils.prompt_choice("choose an option ", ["1", "2"])
        display= display_manager.DisplayManager()

        if instruction_mode == "1":
            display.training_display()
        elif instruction_mode == "2":
            display.identification_display()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        exit()
    