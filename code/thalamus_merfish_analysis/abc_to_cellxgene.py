# This module contains code to export an ABC Atlas dataset to an h5ad file that
# conforms to the CELLxGENE schema. Documentation for the current version of the
# schema, 5.0.0 as of April 2024, can be found at:
# https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/5.0.0/schema.md

import anndata as ad  # v0.8.0 is canonical
