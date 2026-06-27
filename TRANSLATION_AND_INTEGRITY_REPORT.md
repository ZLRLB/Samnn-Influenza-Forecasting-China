# Translation and code-integrity report

## Scope

Eleven Python scripts were converted into public English versions.

The following elements were translated or neutralized:

- Chinese comments and docstrings
- Chinese console messages
- Chinese plot titles, legends, and axis labels
- revision-process labels such as "revision addition", "new", "optimized version", and "corrected version"
- conversational drafting language such as "you already added", "it is recommended", and "change this to"

These phrases are not proof of AI generation. They are editorial and conversational traces that are inappropriate for a public scientific code release. They were replaced with neutral technical descriptions.

## Integrity checks

Each translated script passed Python syntax compilation.

A normalized abstract-syntax-tree comparison was performed between each original script and its English counterpart. After normalizing string literals, all eleven pairs had identical program structure. This confirms that the following were preserved:

- control flow
- function and class structure
- model layers
- feature lists
- numeric constants
- hyperparameters
- forecast horizons
- training and testing dates
- metric calculations
- sampling and plotting operations

Only comments, docstrings, human-readable labels, messages, and selected output filenames were translated.

The legacy Chinese input-data filenames were retained in the executable code to preserve compatibility with the original local workflow.

## Important scope note

The translated scripts were syntax-checked, but an end-to-end numerical rerun was not performed because the prepared analysis datasets and intermediate SHAP summary files were not included in the release package.
