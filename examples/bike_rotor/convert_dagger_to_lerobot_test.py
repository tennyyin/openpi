"""Tests for the pure resampling logic in convert_dagger_to_lerobot.

A DAgger take carries ONE command per 0.4 s AR block and must be written at a fixed 10 Hz,
so most rows are interpolated. These cover the two places interpolation must NOT happen --
the gripper columns (binary commands do not ramp) and a policy<->human boundary (nobody
issued a blended command there) -- plus the segmentation around them. No lerobot dataset is
built and no GPU is touched. Run with the openpi uv env: `uv run pytest examples/bike_rotor/`.
"""

import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "convert_dagger", os.path.join(_HERE, "convert_dagger_to_lerobot.py"))
cd = importlib.util.module_from_spec(_spec)
# Executing the module imports lerobot at top level; if unavailable, skip the file.
try:
    _spec.loader.exec_module(cd)
    _HAVE = True
except Exception:  # noqa: BLE001
    _HAVE = False

import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(not _HAVE, reason="lerobot deps unavailable")

#: The shipped tri_bike checkpoint: 10 fps root, fpb=1 -> 4 RGB frames = 0.4 s per block,
#: so 2.5 Hz of commands resampled up to 10 Hz = 4 output rows per block.
META = {"data_fps": 10.0, "block_hz": 2.5, "rgb_frames_per_block": 4}
OPEN, CLOSED = 0.1, 0.0


def _args(**kw):
    return cd.Args(repo_id="test/dagger", record_root="/nonexistent", **kw)


def _take(B, grip_closes_at=None, source=None):
    """A [B,20] command stream: xyz ramps, rot6d is identity, grippers step once."""
    ac = np.zeros((B, cd.ACTION_DIM), dtype=np.float32)
    ac[:, 0] = np.arange(B) * 0.01                      # right xyz moves 1 cm per block
    ac[:, 9] = np.arange(B) * 0.01
    for s, _e in cd.ACTION_ROT6D_SLICES:
        ac[:, s + 0] = 1.0                              # identity rot6d (x=[1,0,0], y=[0,1,0])
        ac[:, s + 4] = 1.0
    ac[:, list(cd.GRIP_COLS)] = OPEN
    if grip_closes_at is not None:
        ac[grip_closes_at:, list(cd.GRIP_COLS)] = CLOSED
    obs = np.tile(np.arange(B, dtype=np.float32)[:, None], (1, cd.STATE_DIM))
    src = (np.full(B, cd.SOURCE_POLICY, dtype=np.uint8) if source is None
           else np.asarray(source, dtype=np.uint8))
    return obs, ac, src


def _plan(B, *, grip_closes_at=None, source=None, **kw):
    obs, ac, src = _take(B, grip_closes_at, source)
    valid = np.ones(B, dtype=bool)
    keep = np.ones(B, dtype=bool)
    return cd.plan_rollout(obs, ac, keep, valid, src, B * 4, META, _args(**kw))


def test_gripper_is_never_interpolated_across_a_block():
    """The whole point: only OPEN or CLOSED may reach the dataset, never 0.05."""
    segs = _plan(6, grip_closes_at=3)
    assert len(segs) == 1
    _st, ac, _fi = segs[0]
    for col in cd.GRIP_COLS:
        g = ac[:, col].astype(np.float64)
        # np.isclose, not a set of literals: these are float32 commands, so 0.1 is
        # 0.100000001... and an exact-membership check fails on correct output.
        assert np.isclose(g[:, None], [OPEN, CLOSED], atol=1e-6).any(1).all(), g


def test_gripper_steps_at_the_block_boundary_it_was_commanded_on():
    """Held FORWARD from the block that issued it -- 4 rows per block at 2.5 -> 10 Hz."""
    _st, ac, _fi = _plan(6, grip_closes_at=3)[0]
    closed = ac[:, cd.GRIP_COLS[0]] == CLOSED
    assert not closed[:12].any(), "closed early: rows for blocks 0-2 must stay open"
    assert closed[12:].all(), "did not close on the block that commanded it"


def test_the_pose_is_still_interpolated():
    """Gripper-hold must not quietly turn the whole action into --resample hold."""
    _st, ac, _fi = _plan(4)[0]
    x = ac[:, 0]
    assert len(np.unique(x.round(6))) > 4, "xyz collapsed to one value per block"
    assert np.all(np.diff(x) >= -1e-9), "xyz should be monotonic for a monotonic input"


def test_resample_hold_leaves_the_gripper_identical():
    """Under --resample hold the extra assignment is a no-op, not a shift."""
    a = _plan(6, grip_closes_at=3, resample="hold")[0][1]
    b = _plan(6, grip_closes_at=3, resample="hold", gripper_hold=False)[0][1]
    assert np.array_equal(a, b)


def test_gripper_hold_false_reproduces_the_ramp_it_exists_to_prevent():
    """Guards the test itself: without the fix the off-manifold rows really do appear."""
    _st, ac, _fi = _plan(6, grip_closes_at=3, gripper_hold=False)[0]
    mid = ac[:, cd.GRIP_COLS[0]]
    assert ((mid > CLOSED + 1e-6) & (mid < OPEN - 1e-6)).any()


def test_no_row_is_blended_across_a_takeover():
    """A policy->human boundary holds the earlier block: nobody commanded the average."""
    src = [cd.SOURCE_POLICY] * 3 + [cd.SOURCE_HUMAN] * 3
    obs, ac, s = _take(6, None, src)
    valid = np.ones(6, dtype=bool)
    segs = cd.plan_rollout(obs, ac, np.ones(6, dtype=bool), valid, s, 24, META, _args())
    _st, out, _fi = segs[0]
    # Rows attributed to block 2 bracket blocks 2 and 3 -- the seam. They must all equal
    # block 2's command exactly rather than interpolating toward block 3's.
    assert np.allclose(out[8:12], ac[2], atol=1e-6)
