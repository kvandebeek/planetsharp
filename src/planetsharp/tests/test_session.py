import tempfile
import unittest

from planetsharp.core.models import BlockInstance, FilterWeights, Session
from planetsharp.core.presets import DEFAULT_FILTER_PRESETS
from planetsharp.persistence.session_store import SessionStore
from planetsharp.processing.engine import WorkflowEngine


class SessionTests(unittest.TestCase):
    def test_filter_weight_validation(self):
        w = FilterWeights(r=100, g=0, b=33, l=1)
        w.validate()
        with self.assertRaises(ValueError):
            FilterWeights(r=120).validate()

    def test_presets_present(self):
        self.assertIn("HOO", DEFAULT_FILTER_PRESETS)
        self.assertIn("SHO", DEFAULT_FILTER_PRESETS)

    def test_save_reload_roundtrip(self):
        session = Session()
        session.stage1_blocks.append(BlockInstance(type="NOISE", params={"strength": 0.2}))
        session.stage2_blocks.append(BlockInstance(type="SATUR", params={"global_saturation": 1.2}))
        with tempfile.NamedTemporaryFile(suffix=".planetflow.json") as f:
            SessionStore.save(f.name, session)
            loaded = SessionStore.load(f.name)
        self.assertEqual(len(loaded.stage1_blocks), 1)
        self.assertEqual(loaded.stage1_blocks[0].type, "NOISE")
        self.assertEqual(len(loaded.stage2_blocks), 1)
        self.assertEqual(loaded.stage2_blocks[0].type, "SATUR")

    def test_deterministic_render_signal(self):
        session = Session()
        session.stage2_workflow.blocks = [
            BlockInstance(type="CONTR", enabled=True),
            BlockInstance(type="SATUR", enabled=False),
            BlockInstance(type="CURVE", enabled=True),
        ]
        engine = WorkflowEngine()
        a = engine.render(session).final
        b = engine.render(session).final
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
