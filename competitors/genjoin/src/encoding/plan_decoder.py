# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Added for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Addition: a helper to convert model output to a flat hint vector.

import numpy as np


def model_output_to_hints(output, join_graph, norm=100, method='argmax'):
    return output.detach().cpu().numpy().reshape(-1)
