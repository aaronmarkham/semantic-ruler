"""Concept definitions: what a ruler measures and the blocks it is averaged over."""

from __future__ import annotations

import re
from dataclasses import dataclass


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass(frozen=True)
class Block:
    """One domain in the blocked design; its jargon is what the ruler must ignore."""

    name: str
    focus: str

    @property
    def slug(self) -> str:
        return slugify(self.name)


@dataclass(frozen=True)
class Concept:
    name: str
    scale_name: str  # short noun for the graded quantity
    subject: str  # what each sample is
    neg_label: str  # lambda = 0.0
    pos_label: str  # lambda = 1.0
    dimensions: str  # the only aspects allowed to differ within a pair
    context: str
    blocks: tuple[Block, ...]

    @property
    def slug(self) -> str:
        return slugify(self.name)


# Case Study I (Section 4.1).
SUSTAINABILITY = Concept(
    name="Environmentally Conscious Manufacturing",
    scale_name="sustainability",
    subject="manufacturing process description",
    neg_label="Highly UNsustainable (traditional, linear-economy process)",
    pos_label="Highly sustainable (circular, optimized process)",
    dimensions=(
        "sustainability dimensions (energy, materials, waste, emissions, scrap, "
        "monitoring, circularity, logistics)"
    ),
    context="Manufacturing a component for industrial use.",
    blocks=(
        Block(
            "CNC Machining",
            "subtractive manufacturing of aluminum components; coolants, chip management",
        ),
        Block(
            "Injection Molding",
            "polymer processing; regrind usage, cycle times, energy monitoring",
        ),
        Block(
            "Additive Manufacturing",
            "metal powder bed fusion; powder recycling, inert gas usage",
        ),
        Block(
            "Surface Treatment",
            "powder coating and anodizing; chemical recovery, water usage",
        ),
    ),
)

CONCEPTS = {SUSTAINABILITY.slug: SUSTAINABILITY, "sustainability": SUSTAINABILITY}
