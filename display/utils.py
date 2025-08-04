@staticmethod
def prompt_choice(prompt, valid_options):
    while True:
        user_input = input(prompt).strip()
        if user_input in valid_options or user_input is None:
            return user_input
        print(f"Invalid input. Choose from: {', '.join(valid_options)}")
        