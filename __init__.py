import os
import base64

from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from embeddings import (
        run_embeddings_model,
        HIERA_ARCHS,
    )

def _handle_calling(
        uri, 
        sample_collection, 
        model_name,
        emb_field,
        embedding_types, 
        delegate=False
        ):
    ctx = dict(dataset=sample_collection)

    params = dict(
        model_name=model_name,
        emb_field=emb_field,
        embedding_types=embedding_types,
        delegate=delegate
        )
    return foo.execute_operator(uri, ctx, params=params)

class HieraEmbeddings(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            # The operator's URI: f"{plugin_name}/{name}"
            name="compute_hiera_embeddings",  # required

            # The display name of the operator
            label="Compute embeddings with Hiera models",  # required

            # A description for the operator
            description="Compute embeddings using Hiera Models from Facebook/Meta",

            icon="/assets/icons8-meta.svg",

            )

    def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()

        model_dropdown = types.Dropdown(label="Choose the AIMv2 embedding model you want to use:")

        for arch in HIERA_ARCHS:
            model_dropdown.add_choice(arch, label=arch)

        inputs.enum(
            "model_name",
            values=model_dropdown.values(),
            label="Embedding Model",
            description="Select from one of the supported models. Note: The model weights will be downloaded from Hugging Face.",
            view=model_dropdown,
            required=True
        )

        embedding_types = types.RadioGroup(label="Which embedding approach do you want to use?",)

        embedding_types.add_choice(
            "cls", 
            label="Class token embedding",
            description="A single embedding vector derived from special classification token. Represents the global semantic context of an image."
            )
        
        embedding_types.add_choice(
            "mean", 
            label="Mean pooling embedding",
            description="An embedding vector computed by averaging the representations of all image patches. Captures distributed contextual information across the entire input."
            )
        
        inputs.enum(
            "embedding_types",
            values=embedding_types.values(),
            view=embedding_types,
            required=True
        )

        inputs.str(
            "emb_field",            
            required=True,
            description="Name of the field to store the embeddings in."
            )
        
        inputs.bool(
            "delegate",
            default=False,
            required=True,
            label="Delegate execution?",
            description=("If you choose to delegate this operation you must first have a delegated service running. "
            "You can launch a delegated service by running `fiftyone delegated launch` in your terminal"),
            view=types.CheckboxView(),
        )

        inputs.view_target(ctx)

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        """Implement this method if you want to programmatically *force*
        this operation to be delegated or executed immediately.

        Returns:
            whether the operation should be delegated (True), run
            immediately (False), or None to defer to
            `resolve_execution_options()` to specify the available options
        """
        return ctx.params.get("delegate", False)


    def execute(self, ctx):
        """Executes the actual operation based on the hydrated `ctx`.
        All operators must implement this method.

        This method can optionally be implemented as `async`.

        Returns:
            an optional dict of results values
        """
        view = ctx.target_view()
        model_name = ctx.params.get("model_name")
        emb_field = ctx.params.get("emb_field")
        embedding_types = ctx.params.get("embedding_types")
        
        run_embeddings_model(
            dataset=view,
            model_name=model_name,
            emb_field=emb_field,
            embedding_types=embedding_types
            )
        
        ctx.ops.reload_dataset()

    def __call__(
            self, 
            sample_collection, 
            model_name, 
            emb_field,
            embedding_types,
            delegate=False
            ):
        return _handle_calling(
            self.uri,
            sample_collection,
            model_name,
            emb_field,
            embedding_types, 
            delegate=False
            )

def register(p):
    """Always implement this method and register() each operator that your
    plugin defines.
    """
    p.register(HieraEmbeddings)