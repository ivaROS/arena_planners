"""KISS observation pipeline: YAML-declared collectors + (optional) TF generator -> dict.

Each planner ships a YAML naming its sources by ROS message type or by built-in
generator name (currently just RobotPoseTFGenerator). The pipeline subscribes,
preprocesses, and on collect() returns the latest value per source.

Future sensor combination plugs in here as additional Generator subclasses (which
already exist via the Generator base class in data_sources.base).
"""

from __future__ import annotations

import os

from collections.abc import Callable
from typing import Any

from arena_rclpy_mixins import Namespace
from rclpy.node import Node
from rclpy.qos import QoSProfile

from .data_sources.base import Collector, DataSource, Generator
from .data_sources.collectors import (
    ArenaPedestrianCollector,
    LaserScanCollector,
    OdometryCollector,
    PathCollector,
    PeopleCollector,
    PoseStampedCollector,
    TwistCollector,
)
from .data_sources.tf import RobotPoseTFGenerator

_TYPE_REGISTRY: dict[str, Callable[..., DataSource]] = {
    "sensor_msgs/LaserScan": LaserScanCollector,
    "nav_msgs/Odometry": OdometryCollector,
    "nav_msgs/Path": PathCollector,
    "geometry_msgs/PoseStamped": PoseStampedCollector,
    "geometry_msgs/Twist": TwistCollector,
    "people_msgs/People": PeopleCollector,
    "arena_people_msgs/Pedestrians": ArenaPedestrianCollector,
    "RobotPoseTFGenerator": RobotPoseTFGenerator,
}


class Pipeline:
    def __init__(self, node: Node, ns: str | Namespace = "") -> None:
        self._node = node
        self._ns = Namespace(ns)
        self._collectors: dict[str, Collector] = {}
        self._generators: dict[str, Generator] = {}
        self._sub_handles: list = []

    def add(self, name: str, source: DataSource, qos: QoSProfile | int | None = None) -> None:
        if isinstance(source, Collector):
            self._collectors[name] = source
            if source.topic.startswith("/") or not self._ns:
                topic = source.topic
            else:
                # Namespace() joins via os.path.join without resolving "..", so a
                # parent-relative topic (e.g. "../arena_peds" to reach a node-level
                # topic from a robot namespace) would stay literal and be rejected.
                # normpath collapses ".." (and is a no-op for ordinary topics).
                topic = os.path.normpath(str(self._ns(source.topic)))
            qos_value = qos if qos is not None else 10
            self._sub_handles.append(
                self._node.create_subscription(source.message_type, topic, source.update, qos_value)
            )
        elif isinstance(source, Generator):
            self._generators[name] = source
        else:
            raise TypeError(f"unsupported data source type {type(source).__name__}")

    def alias(self, alias: str, target: str) -> None:
        if target in self._collectors:
            self._collectors[alias] = self._collectors[target]
        elif target in self._generators:
            self._generators[alias] = self._generators[target]
        else:
            raise KeyError(f"alias {alias!r} -> unknown target {target!r}")

    def collect(self) -> dict[str, Any]:
        out: dict[str, Any] = {name: c.get_observation() for name, c in self._collectors.items()}
        for name, gen in self._generators.items():
            out[name] = gen.get_observation(out)
        return out

    def shutdown(self) -> None:
        for sub in self._sub_handles:
            self._node.destroy_subscription(sub)
        self._sub_handles.clear()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        node: Node,
        ns: str | Namespace = "",
        source_frame: str = "",
        target_frame: str = "map",
    ) -> Pipeline:
        """Build a pipeline from a YAML-shaped dict.

        Shape:
            aliases: {alias_name: source_name, ...}
            datasources:
              <name>:
                type: <ros_msg_type_string or generator_class_name>
                params:
                  topic: <topic_name>  # for collectors
                  ...                   # generator-specific kwargs
        """
        p = cls(node, ns=ns)
        aliases = config.get("aliases", {}) or {}

        for name, ds_config in (config.get("datasources") or {}).items():
            type_str = ds_config["type"]
            params = dict(ds_config.get("params") or {})
            cls_ = _TYPE_REGISTRY.get(type_str)
            if cls_ is None:
                raise KeyError(f"unknown observation type {type_str!r}; registered: {sorted(_TYPE_REGISTRY)}")

            if issubclass(cls_, Collector):
                topic = params.pop("topic", name)
                source = cls_(name, topic=topic, **params)
            else:
                source = cls_(name, node=node, source_frame=source_frame, target_frame=target_frame, **params)
            p.add(name, source)

        for alias, target in aliases.items():
            p.alias(alias, target)

        return p
