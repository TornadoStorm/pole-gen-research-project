WARNING_COL = "\033[93m"
RESET_COL = "\033[0m"


def warning_format(message, category, filename, lineno, line=None):
    return f"{WARNING_COL}{message}{RESET_COL}\n"
