import numpy as np
from typing import Any, Literal
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def build_rel_axes(
        title: str, 
        class_type: Literal["Binary", "Multi-class"],
        ax: Any
) -> None:
    # Make sure ax is on
    ax.axis("on")
    y_label = "Frequency" if class_type == "Binary" else "Accuracy"
    ax.plot([0, 1], [0, 1], linestyle='dotted', linewidth=3, color='gray', alpha=0.5)
    # Set title and axis labels
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Confidence")
    # Set x and y limits
    ax.set_xlim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim([0, 1]) 
    