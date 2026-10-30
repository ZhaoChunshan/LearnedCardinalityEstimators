from typing import Dict, List
import os
import torch

from model import DACELora
from plan_utils import (
    get_plan_sequence,
    generate_seqs_encoding,
    pad_sequence,
    get_attention_mask,
    get_op_name_to_one_hot,
    add_numerical_scalers,
)
from utils import ROOT_DIR, plan_parameters, load_json


class CostModel:
    MODEL_PATH = "checkpoints/DACE_ours_no_imdb.ckpt"
    STATISTICS_PATH = "checkpoints/feature_statistics.json"

    def __init__(self):
        # Load feature statistics and build encoders/scalers
        statistics_path = os.path.join(ROOT_DIR, self.STATISTICS_PATH)
        self.feature_statistics = load_json(statistics_path)
        add_numerical_scalers(self.feature_statistics)
        self.op_name_to_one_hot = get_op_name_to_one_hot(self.feature_statistics)

        # Model hyperparameters (must match training config)
        self.node_length = 18
        self.pad_length = 26
        self.max_runtime = 30000  # ms, used to normalize runtimes during training
        self.device = "cuda"

        # Load the DACE model
        self.model = DACELora(
            input_dim=self.node_length,
            hidden_dim=128,
            output_dim=1,
        )
        model_path = os.path.join(ROOT_DIR, self.MODEL_PATH)
        checkpoint = torch.load(model_path, map_location="cpu")
        # The checkpoint was saved from PL_DACE which wraps DACELora in self.model,
        # so state_dict keys have a "model." prefix that we need to strip.
        state_dict = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def estimate_cost(self, plan: Dict) -> float:
        """Estimate the cost (execution time in ms) of a single plan.

        Args:
            plan: A PostgreSQL physical plan. Accepts two formats:
                  - [{"Plan": {...}}]  (as returned by pilotscope)
                  - {"Plan": {...}}    (a single plan dict)

        Returns:
            Estimated execution time in milliseconds.
        """
        return self.estimate_cost_batch([plan])[0]

    def estimate_cost_batch(self, plans: List[Dict]) -> List[float]:
        """Estimate costs of many plans in a single batched forward pass.

        Args:
            plans: list of PostgreSQL physical plans, each either
                   [{"Plan": {...}}] or {"Plan": {...}}.

        Returns:
            Estimated execution times (ms), one per input plan, in the same order.
        """
        if not plans:
            return []

        # Encode every plan; each _encode_plan returns unbatched tensors
        seq_encodings, attention_masks = [], []
        for plan in plans:
            if isinstance(plan, list):
                plan = plan[0]
            seq_encoding, attention_mask = self._encode_plan(plan)
            seq_encodings.append(seq_encoding)
            attention_masks.append(attention_mask)

        # Add batch dimension: (B, pad_length * node_length) and (B, pad_length, pad_length)
        seq_encoding = torch.stack(seq_encodings, dim=0).to(self.device)
        attention_mask = torch.stack(attention_masks, dim=0).to(self.device)

        # Run one batched inference
        with torch.no_grad():
            est_times = self.model(seq_encoding, attention_mask)
            root_preds = est_times[:, 0].cpu()  # root node is at position 0

        # Denormalize: model output is sigmoid-normalized by max_runtime
        return (root_preds * self.max_runtime).tolist()

    @staticmethod
    def _add_actual_total_time(plan_node: Dict) -> None:
        """Recursively add a dummy 'Actual Total Time' field to every node.

        The DACE encoding pipeline (dfs/get_plan_sequence) expects this field to
        exist on every node. For inference-time plans (EXPLAIN without ANALYZE)
        it is absent, so we fill in 0 as a placeholder — it is only used as a
        training label and does not affect the model's feature encoding or output.
        """
        if "Actual Total Time" not in plan_node:
            plan_node["Actual Total Time"] = 0.0
        for child in plan_node.get("Plans", []):
            CostModel._add_actual_total_time(child)

    def _encode_plan(self, plan: Dict):
        """Encode a single plan dict into model-ready tensors.

        The plan dict must have a "Plan" key at the top level.
        """
        # get_plan_sequence expects a "database_id" field; add a dummy one for inference
        if "database_id" not in plan:
            plan["database_id"] = 0

        # Ensure every node has "Actual Total Time" so dfs() doesn't KeyError.
        # EXPLAIN (FORMAT JSON) without ANALYZE omits this field.
        CostModel._add_actual_total_time(plan["Plan"])

        # Step 1: DFS traverse to flatten the plan tree
        seq, run_times, adjs, heights, database_id = get_plan_sequence(
            plan, self.pad_length
        )

        # Step 2: Encode each node: one-hot(Node Type) + scaled(Total Cost, Plan Rows)
        seq_encoding = generate_seqs_encoding(
            seq, self.op_name_to_one_hot, plan_parameters, self.feature_statistics
        )

        # Step 3: Pad to fixed length (pad_length nodes)
        seq_encoding, seq_length = pad_sequence(
            seq_encoding,
            padding_value=0,
            node_length=self.node_length,
            max_length=self.pad_length,
        )

        # Step 4: Build tree-structured attention mask from adjacency
        attention_mask = get_attention_mask(
            adjs, seq_length, self.pad_length, self.node_length
        )

        return seq_encoding, attention_mask
