## Facebook Hiera Embeddings Plugins

### Plugin Overview

This plugin allows you to compute embeddings using [Facebook's Hiera](https://github.com/facebookresearch/hiera) models on your FiftyOne datasets.

**Note*:* This plugin only computing supports image embeddings. 

#### Supported Models

This plugin supports all currently released versions of the [Hiera collection]():

- `facebook/hiera-tiny-224-hf`
- `facebook/hiera-small-224-hf`
- `facebook/hiera-base-224-hf`
- `facebook/hiera-base-plus-224-hf`
- `facebook/hiera-large-224-hf`
- `facebook/hiera-huge-224-hf`



## Installation

If you haven't already, install FiftyOne:

```shell
pip install -U fiftyone transformers
```

Then, install the plugin:

```shell
fiftyone plugins download https://github.com/harpreetsahota204/hiera-embeddings-plugin
```

### Embedding Types

The plugin supports two types of embeddings:

- **Class Token Embedding (`cls`)**: A single embedding vector derived from the special classification token. This represents the global semantic context of an image.
  
- **Mean Pooling Embedding (`mean`)**: An embedding vector computed by averaging the representations of all image patches. This captures distributed contextual information across the entire input.

## Usage in FiftyOne App

You can compute Hiera embeddings directly through the FiftyOne App:

1. Launch the FiftyOne App with your dataset
2. Open the "Operators Browser" by clicking on the Operator Browser icon above the sample grid or by typing backtick (`)
3. Type "compute_hiera_embeddings"
4. Configure the following parameters:
   - **Model**: Select one of the supported Hiera models
   - **Embedding Type**: Choose between:
     - `cls` - Class token embedding for global semantic context
     - `mean` - Mean pooling embedding for distributed contextual information
   - **Field Name**: Enter the name for the embeddings field (e.g., "hiera_embeddings")
5. Click "Execute" to compute embeddings for your dataset

The embeddings will be stored in the specified field name and can be used for similarity searches, visualization, or other downstream tasks. 

**Note:** text-image similarity search is not currently supported.

## Operators

### `compute_hiera_embeddings`

This operator computes image embeddings using a Hiera model.

## Operator usage via SDK

Once the plugin has been installed, you can instantiate the operator as follows:

```python
import fiftyone.operators as foo

embedding_operator = foo.get_operator("@harpreetsahota/hiera_embeddings/compute_hiera_embeddings")
```

You can then compute embeddings on your dataset by running the operator with your desired parameters:

```python
# Run the operator on your dataset
embedding_operator(
    dataset,
    model_name="facebook/hiera-tiny-224-hf",  # Choose any supported model
    embedding_type="cls",  # Either "cls" or "mean"
    field_name="hiera_embeddings",  # Name for the embeddings field
)
```

If you're running in a notebook, it's recommended to launch a [Delegated operation](https://docs.voxel51.com/plugins/using_plugins.html#delegated-operations) by running `fiftyone delegated launch` in terminal, then run as follows:

```python
await embedding_operator(
    dataset,
    model_name="facebook/hiera-tiny-224-hf",  # Choose any supported model
    embedding_type="cls",  # Either "cls" or "mean"
    field_name="hiera_embeddings",  # Name for the embeddings field
)
```
# Citation

You can read the paper here.

```bibtex
@article{ryali2023hiera,
  title={Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles},
  author={Ryali, Chaitanya and Hu, Yuan-Ting and Bolya, Daniel and Wei, Chen and Fan, Haoqi and Huang, Po-Yao and Aggarwal, Vaibhav and Chowdhury, Arkabandhu and Poursaeed, Omid and Hoffman, Judy and Malik, Jitendra and Li, Yanghao and Feichtenhofer, Christoph},
  journal={ICML},
  year={2023}
}
```