"""Edge node and subprocess helper for the DRL planner bridge."""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import signal
import subprocess
import threading
import typing
import uuid

import geometry_msgs.msg
import numpy as np
import rclpy.qos
import yaml
from arena_rclpy_mixins import ArenaMixinNode

from .protocol import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    Action,
    Bye,
    Cancel,
    CancelAck,
    Frame,
    Init,
    InitAck,
    Obs,
    ProtocolError,
    Reset,
    ResetAck,
    Shutdown,
    decode_frame,
    encode_frame,
)
from .transport import (
    OBS_POLICY_LATEST_ONLY,
    OBS_POLICY_LOSSLESS,
    TransportPair,
    ZmqPullTransport,
    ZmqPushTransport,
    generate_pair,
)

if typing.TYPE_CHECKING:
    from arena_planners.observations.pipeline import Pipeline


class PlannerProcess:
    """Thin subprocess wrapper owning one planner child process."""

    def __init__(
        self,
        command: list[str],
        endpoints: TransportPair,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._endpoints = endpoints
        self._extra_env = extra_env or {}
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        env = os.environ.copy()
        env["ARENA_PLANNER_OBS_ENDPOINT"] = self._endpoints.obs.endpoint
        env["ARENA_PLANNER_ACTION_ENDPOINT"] = self._endpoints.action.endpoint
        env.update(self._extra_env)
        self._proc = subprocess.Popen(
            self._command,
            start_new_session=True,
            env=env,
        )

    def is_alive(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    @property
    def returncode(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.poll()

    def stop(self, grace_seconds: float = 2.0) -> int:
        if self._proc is None:
            return 0
        if self._proc.poll() is not None:
            return self._proc.returncode  # type: ignore[return-value]
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        try:
            self._proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            self._proc.wait()
        return self._proc.returncode  # type: ignore[return-value]

    def __enter__(self) -> PlannerProcess:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


_CMD_VEL_QOS = rclpy.qos.QoSProfile(
    depth=1,
    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
    durability=rclpy.qos.DurabilityPolicy.VOLATILE,
)


def _load_manifest(manifest: dict | str) -> dict:
    if isinstance(manifest, dict):
        return manifest
    with open(manifest) as fh:
        return yaml.safe_load(fh) or {}


class PlannerEdgeNode(ArenaMixinNode):
    """Edge-side ROS node that owns one planner subprocess and drives the obs/action loop."""

    def __init__(
        self,
        node_name: str,
        manifest: dict | str,
        planner_command: list[str],
        cmd_vel_topic: str,
        namespace: str = "",
        source_frame: str = "",
        target_frame: str = "map",
        **kwargs: object,
    ) -> None:
        super().__init__(node_name, namespace=namespace, use_global_arguments=False, **kwargs)
        self._manifest_raw = manifest
        self._planner_command = planner_command
        self._cmd_vel_topic = cmd_vel_topic
        self._source_frame = source_frame
        self._target_frame = target_frame

        self._obs_manager: Pipeline | None = None
        self._proc: PlannerProcess | None = None
        self._push: ZmqPushTransport | None = None
        self._pull: ZmqPullTransport | None = None
        self._run_id: str = uuid.uuid4().hex

        self._obs_queue: queue.Queue[bytes] = queue.Queue()
        self._action_queue: asyncio.Queue[Frame] = asyncio.Queue()
        self._ack_queue: asyncio.Queue[Frame] = asyncio.Queue()
        self._io_thread: threading.Thread | None = None
        self._io_stop = threading.Event()

        self._cmd_vel_pub: rclpy.publisher.Publisher | None = None
        self._seq: int = 0

        self._rate = self.ROSParam[float]("planner_rate_hz", 10.0)
        # Default raised 0.5 -> 0.8: a converged 5-human SICNav bilevel solve
        # legitimately takes ~0.52s (IPOPT ~0.37 + ~0.15 build/warmstart overhead),
        # so a 0.5s timeout dropped nearly every action to a held cmd_vel one poll
        # late. 0.8 lets the solved action publish immediately for clean 2Hz control.
        # Harmless for fast NN planners (drlvo/crowdnav) whose solves are << 0.5s.
        self._action_timeout = self.ROSParam[float]("planner_action_timeout_s", 0.8)
        self._init_timeout = self.ROSParam[float]("planner_init_timeout_s", 60.0)
        self._dropped_features_logged: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        manifest = _load_manifest(self._manifest_raw)
        obs_config = manifest.get("observations") or {}
        action_type = manifest.get("action_type", "differential_drive")

        endpoints = generate_pair()

        self._cmd_vel_pub = self.create_publisher(
            geometry_msgs.msg.Twist,
            self._cmd_vel_topic,
            _CMD_VEL_QOS,
        )

        from arena_planners.observations.pipeline import Pipeline  # noqa: PLC0415

        ns = self.get_namespace()
        self._obs_manager = Pipeline.from_config(
            config=obs_config,
            node=self,
            ns=ns,
            source_frame=self._source_frame,
            target_frame=self._target_frame,
        )

        self._proc = PlannerProcess(self._planner_command, endpoints)
        self._proc.start()

        # Bind the edge sockets; planner side connects (planner reads obs, edge reads actions).
        # Create with lossless defaults first; reconfigure after init_ack if needed.
        self._push = ZmqPushTransport(
            endpoints.obs.endpoint,
            mode="bind",
            obs_policy=OBS_POLICY_LOSSLESS,
        )
        self._pull = ZmqPullTransport(
            endpoints.action.endpoint,
            mode="bind",
            obs_policy=OBS_POLICY_LOSSLESS,
        )

        init_frame = Init(
            protocol_version=PROTOCOL_VERSION,
            schema_version=SCHEMA_VERSION,
            obs_schema=obs_config,
            action_schema={"action_type": action_type},
            planner_config={},
            run_id=self._run_id,
        )
        self._push.send_frame(encode_frame(init_frame))

        deadline = self.event_loop.time() + self._init_timeout.value
        init_ack_buf: bytes | None = None
        while True:
            if self._pull.poll(timeout_ms=200):
                init_ack_buf = self._pull.recv_frame()
                break
            if not self._proc.is_alive():
                rc = self._proc.returncode
                raise RuntimeError(
                    f"planner subprocess exited before init_ack (returncode={rc}); command={self._planner_command!r}"
                )
            if self.event_loop.time() >= deadline:
                raise TimeoutError(
                    f"planner did not send init_ack within {self._init_timeout.value}s; "
                    f"command={self._planner_command!r}"
                )
            await asyncio.sleep(0)
        init_ack = decode_frame(init_ack_buf)
        if not isinstance(init_ack, InitAck):
            raise ProtocolError(f"expected init_ack, got {init_ack!r}")

        caps: dict = init_ack.capabilities if isinstance(init_ack.capabilities, dict) else {}
        obs_policy_str: str = caps.get("obs_policy", OBS_POLICY_LOSSLESS)

        if obs_policy_str == OBS_POLICY_LATEST_ONLY:
            # Recreate sockets with conflate; the planner has already connected so the
            # new bind will be on the same ipc path after the old socket is closed.
            self._push.close()
            self._pull.close()
            self._push = ZmqPushTransport(
                endpoints.obs.endpoint,
                mode="bind",
                obs_policy=OBS_POLICY_LATEST_ONLY,
            )
            self._pull = ZmqPullTransport(
                endpoints.action.endpoint,
                mode="bind",
                obs_policy=OBS_POLICY_LATEST_ONLY,
            )

        self._io_stop.clear()
        self._io_thread = threading.Thread(
            target=self._io_loop,
            name="planner-io",
            daemon=True,
        )
        self._io_thread.start()

    async def teardown(self) -> None:
        if self._push is not None:
            try:
                self._push.send_frame(encode_frame(Shutdown()))
                if self._pull is not None and self._pull.poll(2000):
                    bye = decode_frame(self._pull.recv_frame())
                    if not isinstance(bye, Bye):
                        self.get_logger().warning(f"expected bye, got {bye!r}")
            except Exception as exc:
                self.get_logger().warning(f"shutdown handshake failed: {exc}")

        self._io_stop.set()
        if self._io_thread is not None:
            self._io_thread.join(timeout=3.0)

        rc: int = 0
        if self._proc is not None:
            rc = self._proc.stop()

        if self._push is not None:
            self._push.close()
        if self._pull is not None:
            self._pull.close()

        if self._obs_manager is not None:
            self._obs_manager.shutdown()

        if rc not in (0, -signal.SIGTERM, -signal.SIGKILL):
            raise RuntimeError(f"planner subprocess exited with code {rc}")

    # ------------------------------------------------------------------
    # I/O thread
    # ------------------------------------------------------------------

    def _io_loop(self) -> None:
        """Dedicated I/O thread: drains _obs_queue out, routes incoming frames in."""
        loop = self.event_loop
        while not self._io_stop.is_set():
            try:
                buf = self._obs_queue.get_nowait()
                if self._push is not None:
                    self._push.send_frame(buf)
            except queue.Empty:
                pass

            if self._pull is not None and self._pull.poll(timeout_ms=10):
                try:
                    frame = decode_frame(self._pull.recv_frame())
                    target = self._action_queue if isinstance(frame, Action) else self._ack_queue
                    loop.call_soon_threadsafe(target.put_nowait, frame)
                except Exception as exc:
                    self.get_logger().error(f"IO thread decode error: {exc}")

            if not self._proc or not self._proc.is_alive():
                self.get_logger().error("planner process died unexpectedly")
                break

    # ------------------------------------------------------------------
    # Per-tick main loop
    # ------------------------------------------------------------------

    async def run_loop(self) -> None:
        """Drive the obs/action cycle. Caller should await this after setup()."""
        assert self._obs_manager is not None

        self.get_logger().info(
            f"run_loop entered rate_hz={self._rate.value} action_timeout_s={self._action_timeout.value}"
        )

        try:
            with self.sim_time_rate(self._rate.value) as (done, rate_events):
                while not done.is_set():
                    await rate_events.get()

                    t = self.sim_time
                    features = self._filter_wire_features(self._obs_manager.collect())
                    self._seq += 1

                    obs_frame = Obs(
                        t_sec=t.sec,
                        t_nanosec=t.nanosec,
                        seq=self._seq,
                        features=features,
                    )
                    self._obs_queue.put_nowait(encode_frame(obs_frame))

                    try:
                        frame = await asyncio.wait_for(
                            self._action_queue.get(),
                            timeout=self._action_timeout.value,
                        )
                    except TimeoutError:
                        self.get_logger().warning(
                            f"no action received within {self._action_timeout.value}s (seq={self._seq})"
                        )
                        continue

                    if isinstance(frame, Action):
                        self._publish_action(frame)
                    else:
                        self.get_logger().warning(
                            f"unexpected non-Action frame on action queue seq={self._seq}: "
                            f"{type(frame).__name__}={frame!r}"
                        )
        except BaseException as exc:
            self.get_logger().error(f"run_loop crashed: {exc!r}")
            raise

    # ------------------------------------------------------------------
    # Outbound control messages
    # ------------------------------------------------------------------

    async def request_cancel(self) -> None:
        """Send Cancel and await CancelAck."""
        assert self._push is not None
        self._push.send_frame(encode_frame(Cancel()))
        frame = await self._drain_until(CancelAck, timeout=5.0)
        if not isinstance(frame, CancelAck):
            raise ProtocolError(f"expected cancel_ack, got {frame!r}")

    async def request_reset(
        self,
        episode_id: str,
        initial_state: dict | None = None,
    ) -> None:
        """Send Reset and await ResetAck; warn if round-trip exceeds one sim step."""
        assert self._push is not None
        t_before = self.sim_time
        self._push.send_frame(encode_frame(Reset(episode_id=episode_id, initial_state=initial_state)))
        frame = await self._drain_until(ResetAck, timeout=5.0)
        if not isinstance(frame, ResetAck):
            raise ProtocolError(f"expected reset_ack, got {frame!r}")
        t_after = self.sim_time
        step_s = 1.0 / self._rate.value
        elapsed = (t_after - t_before).to_seconds()
        if elapsed > step_s:
            self.get_logger().warning(f"reset round-trip {elapsed:.3f}s > one sim step {step_s:.3f}s")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_wire_features(self, features: dict) -> dict:
        """Drop features that aren't msgpack-serializable (raw ROS messages)."""
        out: dict = {}
        for key, value in features.items():
            if value is None or isinstance(value, (bool, int, float, str, bytes, np.ndarray, list, tuple, dict)):
                out[key] = value
                continue
            if key not in self._dropped_features_logged:
                self._dropped_features_logged.add(key)
                self.get_logger().warning(
                    f"dropping non-serializable feature {key!r}: {type(value).__module__}.{type(value).__name__} "
                    "(only emitted on first occurrence per key)"
                )
        return out

    async def _drain_until(self, target: type, timeout: float) -> Frame:
        """Wait for a frame of `target` type from the ack queue, with timeout."""
        deadline = self.event_loop.time() + timeout
        while True:
            remaining = deadline - self.event_loop.time()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {target.__name__}")
            frame = await asyncio.wait_for(self._ack_queue.get(), timeout=remaining)
            if isinstance(frame, target):
                return frame
            self.get_logger().warning(f"discarding unexpected frame while waiting for {target.__name__}: {frame!r}")

    def _publish_action(self, action: Action) -> None:
        if self._cmd_vel_pub is None:
            return
        msg = geometry_msgs.msg.Twist()
        if action.action_type in ("differential_drive",):
            if len(action.action) >= 1:
                msg.linear.x = float(action.action[0])
            if len(action.action) >= 2:
                msg.angular.z = float(action.action[-1])
        elif action.action_type in ("omnidirectional",):
            if len(action.action) >= 1:
                msg.linear.x = float(action.action[0])
            if len(action.action) >= 2:
                msg.linear.y = float(action.action[1])
            if len(action.action) >= 3:
                msg.angular.z = float(action.action[2])
        else:
            self.get_logger().warning(f"unknown action_type {action.action_type!r}")
            return
        self._cmd_vel_pub.publish(msg)
