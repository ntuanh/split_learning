class Colors:
    COLORS = {
        "header": '\033[95m',
        "blue": '\033[94m',
        "green": '\033[92m',
        "yellow": '\033[93m',
        "red": '\033[91m',
        "end": '\033[0m'
    }


def print_with_color(text, color):
    color_code = Colors.COLORS.get(color.lower(), Colors.COLORS["end"])
    print(f"{color_code}{text}{Colors.COLORS['end']}")
