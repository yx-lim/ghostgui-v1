"""Integration coverage for model-owned whole-body motion primitives."""

from __future__ import annotations

import unittest

import numpy as np

from application.ai.edit_session import AIEditSession
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.motion_primitives import build_motion_primitive
from application.ai.motion_services import GhostGUIMotionService
from application.ai.trajectory_edit_spec import (
    TrajectoryEditMode,
    TrajectoryEditSpec,
    TrajectoryOperation,
    TrajectoryOperationType,
)
from application.ai.trajectory_executor import (
    TrajectoryExecutionContext,
    TrajectorySpecExecutor,
)
from application.ai.trajectory_operations import build_trajectory_operation_handlers
from application.ai.trajectory_workflow import CompactMotionRunResult
from application.project_document import ProjectDocument
from core.models import MuJoCoRobotAdapter, RobotStateTimeline


class G1MotionPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.adapter = MuJoCoRobotAdapter("g1")

    def test_burpee_uses_certified_front_down_pose_and_exact_standing_endpoints(self):
        plan = build_motion_primitive(
            self.adapter,
            primitive="burpee",
            duration_seconds=5.0,
        )

        self.assertEqual(plan.variant, "no_jump")
        self.assertEqual([anchor.time_seconds for anchor in plan.anchors], [
            0.0, 0.8, 1.4, 1.8, 2.5, 3.2, 3.6, 4.2, 5.0,
        ])
        np.testing.assert_allclose(plan.anchors[0].qpos, self.adapter.home_qpos)
        np.testing.assert_allclose(plan.anchors[-1].qpos, self.adapter.home_qpos)
        np.testing.assert_allclose(plan.anchors[3].qpos, plan.anchors[4].qpos)
        self.assertIn("certified front-down prone pose", plan.quality_checks)

    def test_local_execution_is_dense_collision_checked_and_reports_concession(self):
        timeline = RobotStateTimeline(self.adapter)
        timeline.set_state(5.0, self.adapter.home_qpos)
        document = ProjectDocument(
            "g1",
            timeline_duration=5.0,
            qpos_timeline=timeline,
        )
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        metadata.seed_document_as_user_owned(document)
        session = AIEditSession(document, metadata_store=store)
        motion = GhostGUIMotionService(self.adapter)
        executor = TrajectorySpecExecutor(
            build_trajectory_operation_handlers(motion, metadata),
            motion.validate_motion,
        )
        spec = TrajectoryEditSpec(
            TrajectoryEditMode.GENERATE,
            "Create a five-second burpee.",
            (TrajectoryOperation(
                TrajectoryOperationType.MOTION_PRIMITIVE,
                {"primitive": "burpee", "duration_seconds": 5.0},
            ),),
        )

        execution = executor.execute(
            spec,
            context=TrajectoryExecutionContext(session, object()),
        )

        generated = session.working_document
        self.assertTrue(execution.validation.valid, execution.validation.issues)
        self.assertEqual(execution.validation.warnings, ())
        self.assertEqual(len(generated.qpos_timeline.times()), 501)
        np.testing.assert_allclose(
            generated.qpos_timeline.get_state(0.0),
            self.adapter.home_qpos,
        )
        np.testing.assert_allclose(
            generated.qpos_timeline.get_state(5.0),
            self.adapter.home_qpos,
        )
        result = CompactMotionRunResult(
            planning=type("Planning", (), {"spec": spec})(),
            execution=execution,
        )
        self.assertTrue(any("no-jump burpee" in line for line in result.proposal_lines))


if __name__ == "__main__":
    unittest.main()
