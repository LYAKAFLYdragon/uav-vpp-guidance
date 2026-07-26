"""Pure F07 canonical VPP action-dimension contract evidence."""
from dataclasses import dataclass

@dataclass(frozen=True)
class CanonicalVPPActionContract:
    dimension: int = 3
    schema_version: str = "vpp-action-contract-1"
    def validate(self, *, config_dim=None, generator_dim=None, action_space_dim=None, checkpoint_dim=None, legacy_compatibility=False):
        values = {"config": config_dim, "generator": generator_dim, "action_space": action_space_dim, "checkpoint": checkpoint_dim}
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(f"F07 canonical VPP action dimension is required; missing {', '.join(missing)}. Silent fallback is prohibited.")
        mismatches = {name: value for name, value in values.items() if int(value) != self.dimension}
        if mismatches:
            mode = "explicit legacy compatibility requested but unsupported" if legacy_compatibility else "compatibility mode was not requested"
            raise ValueError(f"F07 VPP action dimension mismatch: canonical={self.dimension}, observed={mismatches}; {mode}.")
        return {"dimension": self.dimension, "schema_version": self.schema_version, "boundaries": values, "legacy_compatibility": False}

def construction_matrix():
    contract = CanonicalVPPActionContract(); cases = {"compatible_3d": dict(config_dim=3,generator_dim=3,action_space_dim=3,checkpoint_dim=3), "missing_config": dict(config_dim=None,generator_dim=5,action_space_dim=3,checkpoint_dim=3), "generator_mismatch": dict(config_dim=3,generator_dim=5,action_space_dim=3,checkpoint_dim=3), "stale_checkpoint": dict(config_dim=3,generator_dim=3,action_space_dim=3,checkpoint_dim=5)}; results = {}
    for name, inputs in cases.items():
        try: results[name] = {"accepted": True, "contract": contract.validate(**inputs)}
        except ValueError as exc: results[name] = {"accepted": False, "error": str(exc)}
    return {"canonical_dimension": 3, "schema_version": contract.schema_version, "cases": results, "legacy_policy": "reject dimensions other than canonical 3 until a separately versioned compatibility adapter is approved"}
