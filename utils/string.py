def format_accuracy(accuracy: float, color=True) -> str:
    formatted_accuracy = f"{accuracy * 100:.2f}%"
    if not color:
        return formatted_accuracy

    COLOR_GRADIENT = [
        (0.6, (255, 0, 0)),
        (0.8, (255, 255, 0)),
        (1.0, (0, 255, 0)),
    ]

    def lerp_color(value, start, end):
        return tuple(int(start[i] + (end[i] - start[i]) * value) for i in range(3))

    def get_color(accuracy):
        if accuracy <= 0.6:
            return (255, 0, 0)
        for i in range(len(COLOR_GRADIENT) - 1):
            if COLOR_GRADIENT[i][0] < accuracy <= COLOR_GRADIENT[i + 1][0]:
                start_val, start_color = COLOR_GRADIENT[i]
                end_val, end_color = COLOR_GRADIENT[i + 1]
                value = (accuracy - start_val) / (end_val - start_val)
                return lerp_color(value, start_color, end_color)
        return (0, 255, 0)

    color = get_color(accuracy)
    return f"\033[38;2;{color[0]};{color[1]};{color[2]}m{formatted_accuracy}\033[0m"
