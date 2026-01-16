import matplotlib.pyplot as plt
from utils.data_handler import make_path

def save_current_figure(base_name: str, dpi: int = 200) -> str:
    path = make_path("visualizations", base_name, "png")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    return str(path)