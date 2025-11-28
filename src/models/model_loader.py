from typing import Any, List, Tuple


def get_model_and_classes() -> Tuple[Any, List[str]]:
    """
    Placeholder for loading a trained PyTorch model and class names.

    For now, returns (None, []) and is not used.
    Later, this will:
      - load a torch model from disk
      - return the model + list of class names
    """
    model = None
    class_names: List[str] = []
    return model, class_names
