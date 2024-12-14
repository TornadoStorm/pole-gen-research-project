from typing import List

from IPython.display import HTML, display


def show_collapsible_list(title: str, items: List[str]):
    display(
        HTML(
            f"""
            <details>
                <summary>{title}</summary>
                <ul>
                    {"".join(f"<li>{item}</li>" for item in items)}
                </ul>
            </details>
            """
        )
    )
