"""Free-form task text -> coarse intent, so plugins can match semantics rather
than requiring an exact task id.

The curated scene tasks stay authoritative when their id is known; this module
only decides what to do with a user sentence that matched nothing.
"""
from __future__ import annotations

# Ordered by specificity: the first family that matches wins, so "turn right and
# stop" is a pose task and not an approach task just because it says "then stop".
RECOVER = ("recover", "get up", "stand up", "get back up", "on its feet",
           "起身", "站起来", "恢复", "爬起来")
ORBIT = ("orbit", "circle around", "go behind", "side-rear", "walk around",
         "绕", "侧后方", "转到后面")
GAIT = ("rough ground", "uneven", "gravel", "corridor", "narrow", "ramp",
        "不平", "碎石", "走廊", "窄道", "斜坡")
# Note: no bare "stop"/"停止" here. "go to the cube and stop next to it" is an approach
# task whose final condition happens to be stopping, and classifying it as a pose task
# would switch exploration off.
SPIN_NOTE = "spin"  # a bare spin is a rotation request, not an orbit
POSE = ("turn right", "turn left", "turn around", "spin in place", "spin around", "spin", "stand still", "sit",
        "stay there", "wait there", "原地", "转身", "站着", "别动", "右转", "左转")
MANIPULATE = ("kick", "push", "shove", "nudge", "bring", "retrieve", "carry", "deliver",
              "take the", "take it", "put it", "pick", "grab", "move it",
              "put it in", "into the",
              "踢", "推", "捅", "送到", "运到", "带到", "拿给", "放进", "拿", "捡", "抓")
APPROACH = ("go to", "go over", "walk to", "walk towards", "approach", "reach", "find",
            "get to", "move to", "head to", "come here", "come over", "come and",
            "过去", "走向", "走到", "靠近", "找到", "去找", "过来", "来这里",
            "去", "到", "找", "看看")

# Negation: "don't go near the blue cube" must not resolve the blue cube as a goal.
NEGATION = ("don't", "do not", "avoid", "stay away", "keep away", "not go",
            "别", "不要", "不准", "远离", "避免")


def resolve_intent(task_id: str, task_text: str) -> str:
    """One of recover | orbit | gait | pose | manipulate | approach | unknown."""
    text = (task_text or "").lower()
    if any(w in text for w in NEGATION):
        return "avoid"
    # Manipulation is checked before orbit/gait: "push the ball around the pillar into
    # the zone" is a manipulation task that merely routes around an obstacle.
    for name, words in (("recover", RECOVER), ("manipulate", MANIPULATE), ("orbit", ORBIT),
                        ("gait", GAIT), ("pose", POSE), ("approach", APPROACH)):
        if any(w in text for w in words):
            return name
    return "unknown"


def is_blocking_intent(task_id: str, task_text: str) -> bool:
    """True when walking to a waypoint would fight the objective."""
    return resolve_intent(task_id, task_text) in ("recover", "orbit", "gait", "pose")


def pick_target(hits, task_text: str) -> tuple[str, str | None]:
    """Choose which body the sensor should track, from scanned entity hits.

    ``hits`` is what SceneSpec.scan_text returns: [(label, body, kind), ...] with the
    most specific label first. For "kick the ball into the green zone" the longest
    match is the zone, but the thing to act on is the ball, so manipulation intents
    prefer a non-zone entity. Returns (body, matched label or None).
    """
    if not hits:
        return "ball", None
    intent = resolve_intent("", task_text)
    if intent == "avoid":
        # do not track (and therefore chase) the thing the user asked us to avoid
        return "ball", None
    if intent == "manipulate":
        for label, body, kind in hits:
            if kind != "zone":
                return body, label
    label, body, _kind = hits[0]
    return body, label
