"""CVAT XML 1.1 annotation parser used by labeling and supervised training."""

import os
from xml.etree import ElementTree as ET

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)


class CvatAnnotationParser:
    """Parse a CVAT XML 1.1 annotation file.

    Extracts a mapping from image basename to label string.
    When an image has multiple annotated objects, the label is determined by
    the most frequent (majority-vote) class. Ties are broken alphabetically.

    Usage:
        parser = CvatAnnotationParser("annotations.xml")
        label_map = parser.frame_labels   # {"frame_0000.jpg": "car", ...}
        labels = parser.label_names       # ["bicycle", "bus", ...] (ordered)
    """

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.label_names: list[str] = []
        self.frame_labels: dict[str, str] = {}
        self._parse()

    def _parse(self) -> None:
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        label_order: list[str] = []
        for lbl_el in root.findall("./meta/task/labels/label"):
            name_el = lbl_el.find("name")
            if name_el is not None and name_el.text:
                label_order.append(name_el.text.strip())

        if not label_order:
            seen: dict[str, int] = {}
            for img_el in root.findall("image"):
                for box in img_el.findall("box"):
                    lbl = box.get("label", "").strip()
                    if lbl and lbl not in seen:
                        seen[lbl] = len(seen)
            label_order = sorted(seen.keys())

        self.label_names = label_order

        for img_el in root.findall("image"):
            name_attr = img_el.get("name", "")
            basename = os.path.basename(name_attr)
            if not basename:
                continue

            counts: dict[str, int] = {}
            for box in img_el.findall("box"):
                lbl = box.get("label", "").strip()
                if lbl:
                    counts[lbl] = counts.get(lbl, 0) + 1
            for poly in img_el.findall("polygon"):
                lbl = poly.get("label", "").strip()
                if lbl:
                    counts[lbl] = counts.get(lbl, 0) + 1
            for pts in img_el.findall("points"):
                lbl = pts.get("label", "").strip()
                if lbl:
                    counts[lbl] = counts.get(lbl, 0) + 1

            if not counts:
                continue

            majority_label = max(counts, key=lambda k: (counts[k], k))
            self.frame_labels[basename] = majority_label

        logger.info(
            "CvatAnnotationParser: xml=%s labels=%s frames=%d",
            self.xml_path,
            self.label_names,
            len(self.frame_labels),
        )

    def label_to_idx(self) -> dict[str, int]:
        """Return {label_name: int_index} mapping."""
        return {name: i for i, name in enumerate(self.label_names)}
