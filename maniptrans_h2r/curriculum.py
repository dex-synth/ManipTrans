"""Native residual reset schedule, applied only to the private training world."""

import numpy as np


class ResetSchedule:
    """VecTask frequency gate and const_scale / linear_decay / friction buckets."""

    def __init__(self, count, friction):
        self.nominal = np.asarray(friction, dtype=np.float64).copy()
        self.friction = np.broadcast_to(self.nominal, (count, len(self.nominal))).copy()
        self.age = np.zeros(count, dtype=np.int64)
        self.first = True
        self.last_step = 0
        self.gravity_scale = 1.

    def advance(self):
        self.age += 1

    def reset(self, ids, step):
        update = self.first or step-self.last_step >= 32
        if self.first:
            eligible = np.arange(len(self.age))
        else:
            eligible = np.asarray(ids)[self.age[ids] >= 32]
            # Native consumes eligible ages even when the global gate is closed.
            self.age[eligible] = 0
        if update:
            self.last_step = step
            decay = 1 - (1 / 1920) * min(step, 1920)
            self.gravity_scale = 1-decay
            values = self.nominal*(3*decay + 1*(1-decay))
            buckets = np.asarray([5*i/250+1 for i in range(250)])
            self.friction[eligible] = buckets[np.searchsorted(buckets, values, side="right").clip(1, 250)-1]
        self.first = False

    def state_dict(self):
        return dict(age=self.age.tolist(), first=self.first, last_step=self.last_step,
                    gravity_scale=self.gravity_scale, friction=self.friction.tolist())

    def load_state_dict(self, state):
        self.age[:] = state["age"]
        self.first, self.last_step = state["first"], state["last_step"]
        self.gravity_scale = state["gravity_scale"]
        self.friction[:] = state["friction"]


class PhysicsCurriculum(ResetSchedule):
    def __init__(self, env):
        import torch
        import warp as wp
        self.env = env
        model = env.cpu
        bodies = {model.body(b).id for b in env.metadata["object_bodies"]}
        for body in range(1, model.nbody):
            if model.body_parentid[body] in bodies:
                bodies.add(body)
        self.geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, list(bodies)))
        super().__init__(env.count, model.geom_friction[self.geom_ids, 0])
        self.nominal_gravity = model.opt.gravity.copy()
        # mjlab expands the field per world and rebuilds CUDA graphs before views
        # are acquired. Subsequent in-place copies keep captured pointers valid.
        env.sim.expand_model_fields(("geom_friction",))
        self.gpu_friction = wp.to_torch(env.model.geom_friction)
        self.gpu_gravity = wp.to_torch(env.model.opt.gravity)
        self.geom_indices = torch.as_tensor(self.geom_ids, device=env.device)

    def apply(self):
        import torch
        self.gpu_friction[:, self.geom_indices, 0] = torch.as_tensor(
            self.friction, dtype=self.gpu_friction.dtype, device=self.env.device)
        gravity = self.nominal_gravity*self.gravity_scale
        self.gpu_gravity.copy_(torch.as_tensor(gravity, dtype=self.gpu_gravity.dtype, device=self.env.device))
        self.env.features.gravity = float(np.linalg.norm(gravity))

    def reset(self, ids, step):
        super().reset(ids, step)
        self.apply()

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.apply()

    def audit(self):
        from .physics import equal, gpu_audit
        friction = self.gpu_friction.cpu().numpy()
        expected = np.broadcast_to(self.env.cpu.geom_friction, friction.shape).copy()
        expected[:, self.geom_ids, 0] = self.friction
        equal(friction, expected, "curriculum.geom_friction")
        gravity = self.gpu_gravity.cpu().numpy()
        equal(gravity, np.broadcast_to(self.nominal_gravity*self.gravity_scale, gravity.shape), "curriculum.gravity")
        # Audit all other fields against the sealed nominal model as well.
        import torch
        saved_friction = self.gpu_friction.clone()
        saved_gravity = self.gpu_gravity.clone()
        try:
            self.gpu_friction.copy_(torch.as_tensor(self.env.cpu.geom_friction, device=self.env.device))
            self.gpu_gravity.copy_(torch.as_tensor(self.nominal_gravity, device=self.env.device))
            gpu_audit(self.env.cpu, self.env.model, self.env.task.task_id, self.env.metadata["scene_sha256"])
        finally:
            self.gpu_friction.copy_(saved_friction)
            self.gpu_gravity.copy_(saved_gravity)
        return dict(schema="native-residual-v1", object_geoms=[self.env.cpu.geom(g).name for g in self.geom_ids],
                    state=self.state_dict(), gravity=gravity.tolist(), gpu_readback_matches=True,
                    other_fields_nominal=True)
