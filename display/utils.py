@staticmethod
def prompt_choice(prompt, valid_options):
    while True:
        user_input = input(prompt).strip()
        if user_input in valid_options:
            return user_input
        print(f"Invalid input. Choose from: {', '.join(valid_options)}")