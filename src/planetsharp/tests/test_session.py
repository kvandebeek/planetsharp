import tempfile
import unittest

from planetsharp.core.models import BlockInstance, FilterWeights, Session
from planetsharp.core.presets import DEFAULT_FILTER_PRESETS
from planetsharp.persistence.session_store import SessionStore
from planetsharp.persistence.template_store import TemplateStore
from planetsharp.processing.engine import WorkflowEngine
from planetsharp.io.formats import detect_format


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


    def test_template_roundtrip(self):
        session = Session()
        session.stage1_blocks.append(BlockInstance(type="NOISE", enabled=False, params={"strength": 0.4}))
        session.stage2_blocks.append(BlockInstance(type="SATUR", enabled=True, params={"global_saturation": 1.1}))
        with tempfile.NamedTemporaryFile(suffix=".planetsharp-template.json") as f:
            TemplateStore.save(f.name, session)
            loaded = TemplateStore.load(f.name)
        self.assertEqual([b.type for b in loaded["stage1"]], ["NOISE"])
        self.assertFalse(loaded["stage1"][0].enabled)
        self.assertEqual(loaded["stage2"][0].params["global_saturation"], 1.1)

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

    def test_template_preserves_stage1_channel(self):
        session = Session()
        block = BlockInstance(type="NOISE", enabled=True, params={"strength": 0.4}, channel="R")
        session.stage1_workflows["R"].blocks.append(block)
        with tempfile.NamedTemporaryFile(suffix=".planetsharp-template.json") as f:
            TemplateStore.save(f.name, session)
            loaded = TemplateStore.load(f.name)
        self.assertEqual(loaded["stage1"][0].channel, "R")

    def test_jpg_disabled(self):
        with self.assertRaises(ValueError):
            detect_format("foo.jpg")
        with self.assertRaises(ValueError):
            detect_format("foo.jpeg")


if __name__ == "__main__":
    unittest.main()
