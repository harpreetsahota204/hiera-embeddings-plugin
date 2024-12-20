"""Input resolution utilities for AIMv2 embedding operator."""

from fiftyone.operators import types

def resolve_model_input(inputs, models, config):
    """Resolves the model selection dropdown input."""
    model_dropdown = types.Dropdown(label=config["model"]["dropdown_label"])
    
    for model in config["embedding_models"]:
        model_dropdown.add_choice(model, label=model)
    
    inputs.enum(
        config["model"]["field"],
        values=model_dropdown.values(),
        label=config["model"]["label"],
        description=config["model"]["description"],
        view=model_dropdown,
        required=config["model"]["required"]
    )

def resolve_embedding_types(inputs, config):
    """Resolves the embedding type selection radio group input."""
    embedding_types = types.RadioGroup(label=config["embedding_types"]["label"])
    
    for key, choice in config["embedding_types"]["choices"].items():
        embedding_types.add_choice(
            key,
            label=choice["label"],
            description=choice["description"]
        )
    
    inputs.enum(
        config["embedding_types"]["field"],
        values=embedding_types.values(),
        view=embedding_types,
        required=config["embedding_types"]["required"]
    )

def resolve_embedding_field(inputs, config):
    """Resolves the embedding field name input."""
    inputs.str(
        config["embedding_field"]["field"],
        required=config["embedding_field"]["required"],
        description=config["embedding_field"]["description"]
    )

def resolve_delegate_input(inputs, config):
    """Resolves the delegation checkbox input."""
    inputs.bool(
        config["delegate"]["field"],
        default=config["delegate"]["default"],
        required=config["delegate"]["required"],
        label=config["delegate"]["label"],
        description=config["delegate"]["description"],
        view=types.CheckboxView(),
    )