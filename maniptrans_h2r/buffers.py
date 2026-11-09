"""MJWarp allocation limits and actionable overflow reports."""

import torch
import warp as wp
from mujoco_warp._src.types import OverflowType


CAPACITY_KEYS = ("contact_capacity", "constraint_capacity")


def capacities(recipe):
    result = {key: recipe[key] for key in CAPACITY_KEYS}
    if any(type(value) is not int or value < 1 for value in result.values()):
        raise ValueError("Contact and constraint capacities must be positive integers")
    return result


class SimulationOverflow(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"MJWarp overflow during {report['phase']}: {report['flags']}; "
                         f"{report['world_count']} affected worlds; limits={report['limits']}")


def check_overflow(data, phase):
    masks = wp.to_torch(data.overflow)
    if not torch.any(masks != 0):
        return
    masks = masks.cpu().to(torch.int64)
    ids = masks.nonzero().flatten()
    known = sum(int(flag) for flag in OverflowType)
    flags = {flag.name: int(((masks & int(flag)) != 0).sum())
             for flag in OverflowType if torch.any(masks & int(flag))}
    unknown = masks & ~known
    if unknown.any():
        flags["UNKNOWN"] = int((unknown != 0).sum())
    sample = ids[:16]
    nefc = wp.to_torch(data.nefc)
    raise SimulationOverflow(dict(phase=phase, flags=flags, world_count=len(ids),
        world_ids=sample.tolist(), masks=masks[sample].tolist(),
        nefc=nefc[sample.to(nefc.device)].cpu().tolist(),
        contacts_total=int(wp.to_torch(data.nacon).sum()),
        limits=dict(contacts_total=data.naconmax, constraints_per_world=data.njmax,
                    ccd_total=data.naccdmax, jacobian_nonzeros=data.njmax_nnz)))
