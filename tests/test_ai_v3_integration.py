"""Credential-free integration tests for the complete v3.0 AI edit flow."""

from __future__ import annotations

import asyncio
import json
import unittest

from application.ai import (
    AIEditSession,
    AIEditSessionState,
    EditorSelectionContext,
    GhostGUIAgent,
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    SemanticToolContext,
    TimestampMotionIdentityResolver,
    build_semantic_tool_registry,
    sample_working_preview_qpos,
)
from application.ai.errors import ProviderCancelledError, ProviderError
from application.ai.motion_services import (
    LogicalFrameSolveResult,
    MotionValidationReport,
)
from application.ai.providers import MockProvider, MockStep
from application.ai.schemas import (
    EditAuthor,
    MessageRole,
    ProviderResponse,
    StopReason,
    ToolCall,
)
from application.editor_commands import UpdateKeyframe
from application.editor_controller import EditorController
from application.editor_events import DocumentChanged, EditorEventBus
from application.project_document import ProjectDocument
from core.trajectory import TargetFrame


class _Timeline:
    def __init__(self, states=()):
        self.states = {}
        for time, qpos in states:
            self.set_state(time, qpos)

    def set_state(self, time, qpos):
        self.states[round(float(time), 6)] = list(qpos)

    def get_state(self, time):
        value = self.states.get(round(float(time), 6))
        return None if value is None else value.copy()

    def times(self):
        return sorted(self.states)

    def sample_state(self, time):
        key = round(float(time), 6)
        if key in self.states:
            return self.states[key].copy()
        lower = max((item for item in self.states if item < key), default=None)
        upper = min((item for item in self.states if item > key), default=None)
        if lower is None:
            return self.states[upper].copy()
        if upper is None:
            return self.states[lower].copy()
        fraction = (key - lower) / (upper - lower)
        return [
            start + fraction * (end - start)
            for start, end in zip(self.states[lower], self.states[upper])
        ]


class _MotionService:
    logical_frames = (
        "pelvis",
        "torso",
        "left_foot",
        "right_foot",
        "right_hand",
    )
    end_effectors = ("left_foot", "right_foot", "right_hand")
    joint_names = ("waist_pitch", "right_shoulder")
    joint_groups = {"upper_body": joint_names}

    def __init__(self):
        self.solve_calls = []

    def solve_logical_frame_target(
        self,
        document,
        *,
        logical_frame,
        time_seconds,
        position_m,
        orientation_rpy_rad,
        mode,
        protected_logical_frames,
    ):
        self.solve_calls.append({
            "logical_frame": logical_frame,
            "time_seconds": float(time_seconds),
            "mode": mode,
            "protected": tuple(protected_logical_frames),
        })
        existing = document.trajectory.targets_at_time(time_seconds).get(
            logical_frame
        )
        start = (
            (0.0, 0.0, 0.0)
            if existing is None
            else (existing.x, existing.y, existing.z)
        )
        position = tuple(float(value) for value in position_m)
        if mode == "delta":
            position = tuple(
                start[index] + position[index] for index in range(3)
            )
        orientation = orientation_rpy_rad or (0.0, 0.0, 0.0)
        return LogicalFrameSolveResult(
            frame=TargetFrame(
                time=float(time_seconds),
                phase="ai_edit",
                frame_name=logical_frame,
                x=position[0],
                y=position[1],
                z=position[2],
                roll=orientation[0],
                pitch=orientation[1],
                yaw=orientation[2],
            ),
            qpos=[position[2], float(time_seconds)],
            status="integration IK solved",
        )

    def set_joint_angles(
        self,
        document,
        *,
        time_seconds,
        values,
        protected_logical_frames,
    ):
        qpos = document.qpos_timeline.sample_state(time_seconds)
        for index, name in enumerate(self.joint_names):
            if name in values:
                qpos[index] = float(values[name])
        return qpos

    def ensure_qpos_keyframe(self, document, *, time_seconds):
        return document.qpos_timeline.sample_state(time_seconds)

    def validate_motion(self, document):
        return MotionValidationReport(True, ())


class _Token:
    cancellation_requested = False


def _document(*, duration=6.0):
    states = tuple(
        value
        for value in (
            (0.0, [0.0, 0.0]),
            (2.0, [0.2, 0.2]),
            (3.0, [0.3, 0.3]),
            (5.0, [0.5, 0.5]),
        )
        if value[0] <= duration
    )
    document = ProjectDocument(
        "g1",
        timeline_duration=duration,
        qpos_timeline=_Timeline(states),
    )
    frames = (
        TargetFrame(time=0.0, frame_name="pelvis", z=0.9),
        TargetFrame(time=0.0, frame_name="right_hand", x=0.3, z=1.0),
        TargetFrame(time=2.0, frame_name="pelvis", z=0.75),
        TargetFrame(time=3.0, frame_name="pelvis", z=0.9),
        TargetFrame(time=5.0, frame_name="pelvis", z=0.95),
    )
    for frame in frames:
        if frame.time <= duration:
            document.trajectory.add_frame(frame)
    return document


def _tool_response(*calls):
    return ProviderResponse(
        tool_calls=tuple(
            ToolCall(identifier, name, arguments)
            for identifier, name, arguments in calls
        ),
        stop_reason=StopReason.TOOL_CALLS,
    )


class _Workflow:
    def __init__(
        self,
        responses,
        *,
        document=None,
        selection=None,
        store=None,
        motion=None,
    ):
        self.committed = document or _document()
        self.store = store or InMemoryMotionMetadataStore()
        self.resolver = TimestampMotionIdentityResolver()
        self.metadata = MotionMetadataService(self.store, self.resolver)
        self.session = AIEditSession(
            self.committed,
            metadata_store=self.store,
        )
        self.context = SemanticToolContext(
            self.session,
            self.metadata,
            selection=selection or EditorSelectionContext(),
            motion_name="v3 integration motion",
        )
        self.motion = motion or _MotionService()
        self.tools = build_semantic_tool_registry(self.motion)
        self.provider = MockProvider(responses)
        self.agent = GhostGUIAgent(self.provider, self.tools)

    async def run(self, instruction):
        return await self.agent.run(
            instruction,
            model="mock-v3",
            context=self.context,
        )


class V3AIIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mock_tool_pipeline_stages_orange_preview_without_commit(self):
        flow = _Workflow([
            _tool_response((
                "hand-up",
                "move_end_effector",
                {
                    "end_effector": "right_hand",
                    "time_seconds": 0.0,
                    "delta_m": [0.0, 0.0, 0.1],
                },
            )),
            ProviderResponse(text="Raised the right hand by 10 cm."),
        ])
        committed_before = flow.committed.qpos_timeline.get_state(0.0)

        result = await flow.run("Raise the right hand by 10 cm.")
        preview_qpos = sample_working_preview_qpos(flow.session, 0.0)

        self.assertEqual(result.validation, {"valid": True, "issues": []})
        self.assertEqual(flow.session.state, AIEditSessionState.STAGED)
        self.assertEqual(preview_qpos, [1.1, 0.0])
        self.assertEqual(
            flow.committed.qpos_timeline.get_state(0.0),
            committed_before,
        )
        committed_hand = flow.committed.trajectory.targets_at_time(0.0)[
            "right_hand"
        ]
        self.assertEqual(committed_hand.z, 1.0)
        self.assertEqual(flow.motion.solve_calls[0]["mode"], "delta")

    async def test_selected_interval_is_in_context_and_only_it_is_retimed(self):
        selection = EditorSelectionContext(time_interval=(2.0, 3.0))
        flow = _Workflow([
            _tool_response((
                "slow-section",
                "retime_segment",
                {
                    "start_time_seconds": 2.0,
                    "end_time_seconds": 3.0,
                    "speed": 0.5,
                },
            )),
            ProviderResponse(text="The selected section is twice as slow."),
        ], selection=selection)

        await flow.run("Make this section twice as slow.")

        prompt = flow.provider.requests[0].messages[-1].text
        self.assertIn('"time_interval_seconds":[2.0,3.0]', prompt)
        working_times = sorted(
            frame.time
            for frame in flow.session.working_document.trajectory.frames
            if frame.frame_name == "pelvis"
        )
        self.assertEqual(working_times, [0.0, 2.0, 4.0, 5.0])
        self.assertEqual(
            [frame.time for frame in flow.committed.trajectory.frames
             if frame.frame_name == "pelvis"],
            [0.0, 2.0, 3.0, 5.0],
        )
        untouched = flow.session.working_document.trajectory.targets_at_time(5.0)[
            "pelvis"
        ]
        self.assertEqual(untouched.z, 0.95)

    async def test_user_fixed_feet_are_preserved_by_upper_body_edit(self):
        document = _document()
        for frame in (
            TargetFrame(time=2.0, frame_name="left_foot", x=-0.1),
            TargetFrame(time=2.0, frame_name="right_foot", x=0.1),
            TargetFrame(time=2.0, frame_name="torso", z=1.1),
        ):
            document.trajectory.add_frame(frame)
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        for name in ("left_foot", "right_foot"):
            foot = document.trajectory.targets_at_time(2.0)[name]
            reference = metadata.reference_for_keyframe(foot)
            store.record(reference, EditAuthor.USER)
            store.set_protected(reference, True)
        motion = _MotionService()
        flow = _Workflow([
            _tool_response((
                "torso",
                "set_logical_frame_target",
                {
                    "logical_frame": "torso",
                    "time_seconds": 2.0,
                    "position_m": [0.0, 0.0, 1.2],
                    "mode": "absolute",
                },
            )),
            ProviderResponse(text="Upper body changed; both feet remain fixed."),
        ], document=document, store=store, motion=motion)

        await flow.run(
            "Make the upper body more expressive, but don't change either foot."
        )

        self.assertEqual(
            set(motion.solve_calls[0]["protected"]),
            {"left_foot", "right_foot"},
        )
        before = document.trajectory.targets_at_time(2.0)
        after = flow.session.working_document.trajectory.targets_at_time(2.0)
        for name in ("left_foot", "right_foot"):
            self.assertEqual(after[name].to_dict(), before[name].to_dict())
        self.assertEqual(after["torso"].z, 1.2)
        context_payload = json.loads(
            flow.provider.requests[0].messages[-1].text.split(
                "GhostGUI context:\n", 1
            )[1].split("\n\nUser instruction:", 1)[0]
        )
        protected = {
            item["logical_frame"]
            for item in context_payload["constraints"]["keyframes"]
            if item["protected"]
        }
        self.assertEqual(protected, {"left_foot", "right_foot"})

    async def test_multi_tool_motion_plan_creates_editable_keyframes_via_ik(self):
        calls = (
            (
                "crouch",
                "set_logical_frame_target",
                {
                    "logical_frame": "pelvis",
                    "time_seconds": 0.0,
                    "position_m": [0.0, 0.0, 0.65],
                    "mode": "absolute",
                },
            ),
            (
                "stand",
                "set_logical_frame_target",
                {
                    "logical_frame": "pelvis",
                    "time_seconds": 2.0,
                    "position_m": [0.0, 0.0, 0.95],
                    "mode": "absolute",
                },
            ),
            (
                "wave-up",
                "set_logical_frame_target",
                {
                    "logical_frame": "right_hand",
                    "time_seconds": 3.0,
                    "position_m": [0.3, 0.0, 1.35],
                    "mode": "absolute",
                },
            ),
            (
                "wave-out",
                "set_logical_frame_target",
                {
                    "logical_frame": "right_hand",
                    "time_seconds": 4.0,
                    "position_m": [0.5, 0.0, 1.25],
                    "mode": "absolute",
                },
            ),
        )
        flow = _Workflow([
            _tool_response(*calls),
            ProviderResponse(text="Created an editable crouch, stand, and wave."),
        ], document=_document(duration=4.0))

        await flow.run(
            "Create a four-second motion where G1 crouches, stands up, then waves."
        )

        self.assertEqual(len(flow.motion.solve_calls), 4)
        working = flow.session.working_document
        self.assertEqual(working.qpos_timeline.times(), [0.0, 2.0, 3.0, 4.0])
        self.assertEqual(
            working.trajectory.targets_at_time(4.0)["right_hand"].x,
            0.5,
        )
        self.assertFalse(
            {definition.name for definition in flow.tools.definitions()}
            & {"set_qpos_trajectory", "run_code", "execute_python"}
        )
        self.assertEqual(flow.committed.timeline_duration, 4.0)
        self.assertNotIn(
            "right_hand",
            flow.committed.trajectory.targets_at_time(4.0),
        )

    async def test_accept_is_atomic_and_reject_discards_complete_candidate(self):
        accept_flow = _Workflow([
            _tool_response((
                "pelvis",
                "set_logical_frame_target",
                {
                    "logical_frame": "pelvis",
                    "time_seconds": 0.0,
                    "position_m": [0.0, 0.0, 0.7],
                    "mode": "absolute",
                },
            )),
            ProviderResponse(text="Pelvis edit staged."),
        ])
        await accept_flow.run("Lower the pelvis.")
        events = []
        bus = EditorEventBus()
        bus.subscribe(DocumentChanged, events.append)

        accept_flow.session.accept(EditorController(accept_flow.committed, bus))

        self.assertEqual(accept_flow.committed.revision, 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].operation, "replace_motion_state")
        self.assertEqual(
            accept_flow.committed.trajectory.targets_at_time(0.0)["pelvis"].z,
            0.7,
        )

        reject_flow = _Workflow([
            _tool_response((
                "pelvis",
                "set_logical_frame_target",
                {
                    "logical_frame": "pelvis",
                    "time_seconds": 0.0,
                    "position_m": [0.0, 0.0, 0.6],
                    "mode": "absolute",
                },
            )),
            ProviderResponse(text="Another edit staged."),
        ])
        await reject_flow.run("Lower the pelvis more.")
        reject_flow.session.reject()
        self.assertEqual(reject_flow.committed.revision, 0)
        self.assertEqual(
            reject_flow.committed.trajectory.targets_at_time(0.0)["pelvis"].z,
            0.9,
        )

    async def test_manual_edit_then_refine_uses_and_preserves_working_copy(self):
        flow = _Workflow([
            _tool_response((
                "initial",
                "set_logical_frame_target",
                {
                    "logical_frame": "pelvis",
                    "time_seconds": 0.0,
                    "position_m": [0.0, 0.0, 0.8],
                    "mode": "absolute",
                },
            )),
            ProviderResponse(text="Initial result staged."),
            _tool_response((
                "refine-hand",
                "move_end_effector",
                {
                    "end_effector": "right_hand",
                    "time_seconds": 0.0,
                    "delta_m": [0.0, 0.0, 0.05],
                },
            )),
            ProviderResponse(text="Refined from the manually adjusted result."),
        ])
        await flow.run("Lower the pelvis.")
        pelvis = next(
            frame
            for frame in flow.session.working_document.trajectory.frames
            if frame.frame_name == "pelvis" and frame.time == 0.0
        )
        reference = flow.context.working_metadata.reference_for_keyframe(pelvis)
        index = flow.session.working_document.trajectory.index_of_frame(pelvis)
        flow.session.apply_manual(
            UpdateKeyframe(
                index,
                TargetFrame(time=0.0, frame_name="pelvis", z=0.76),
            ),
            affected_entities=(reference,),
        )

        await flow.run("Keep that pelvis height and raise the hand slightly.")

        refine_prompt = flow.provider.requests[2].messages[-1].text
        self.assertIn('"author":"user"', refine_prompt)
        self.assertEqual(
            flow.session.working_document.trajectory.targets_at_time(0.0)[
                "pelvis"
            ].z,
            0.76,
        )
        flow.session.accept(EditorController(flow.committed))
        self.assertEqual(
            flow.committed.trajectory.targets_at_time(0.0)["pelvis"].z,
            0.76,
        )

    async def test_cancel_failure_and_unknown_tool_leave_committed_motion_safe(self):
        cancellation_flow = _Workflow([
            MockStep(
                response=ProviderResponse(text="late"),
                delay_seconds=0.08,
            )
        ])
        token = _Token()
        task = asyncio.create_task(
            cancellation_flow.agent.run(
                "Move the hand.",
                model="mock-v3",
                context=cancellation_flow.context,
                cancellation_token=token,
            )
        )
        await asyncio.sleep(0.02)
        token.cancellation_requested = True
        with self.assertRaises(ProviderCancelledError):
            await task
        self.assertEqual(cancellation_flow.session.state, AIEditSessionState.READY)
        self.assertEqual(cancellation_flow.committed.revision, 0)

        failure_flow = _Workflow([MockStep(error=RuntimeError("offline"))])
        with self.assertRaises(ProviderError):
            await failure_flow.run("Move the hand.")
        self.assertEqual(failure_flow.session.state, AIEditSessionState.READY)
        self.assertEqual(failure_flow.committed.revision, 0)

        malicious_flow = _Workflow([
            _tool_response((
                "unsafe",
                "run_code",
                {"code": "import os; os.system('unsafe')"},
            )),
            ProviderResponse(text="The unsupported request was rejected."),
        ])
        result = await malicious_flow.run("Run arbitrary Python.")
        self.assertFalse(result.tool_executions[0].succeeded)
        tool_result = malicious_flow.provider.requests[1].messages[-1]
        self.assertEqual(tool_result.role, MessageRole.TOOL)
        self.assertTrue(tool_result.tool_results[0].is_error)
        self.assertEqual(malicious_flow.committed.revision, 0)
        self.assertFalse(malicious_flow.session.has_changes)


if __name__ == "__main__":
    unittest.main()
