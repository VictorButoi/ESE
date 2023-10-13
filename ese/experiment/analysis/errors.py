from typing import List
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_regionsize_distribution(subject_list: List[dict]) -> None:


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_accuracy_vs_labels(subject_list: List[dict]) -> None:


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_accuracy_vs_regionsize(subject_list: List[dict]) -> None:


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_accuracy_vs_boundarydist(subject_list: List[dict]) -> None:
