import mmap
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from rapidfuzz import fuzz


class MonteCarloEvidenceSampling:
    """
    Identifies Regions of Interest (ROI) in large files using Monte Carlo Importance Sampling with given evidence snippets.
    """

    def __init__(
        self, file_path: Union[str, Path], sample_size: int = 50, max_scan: int = 20000
    ):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.sample_size = sample_size  # Number of Monte Carlo sampling iterations
        self.max_scan = max_scan
        self.file_size = self.file_path.stat().st_size
        self.f = open(file_path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_weighted_anchors(self, evidence_list: List[str]) -> Dict[bytes, float]:
        """
        Calculates anchor weights. Logic: shorter anchors or those appearing
        frequently across evidence snippets receive lower weights.

        Args:
            evidence_list: Snippets to guide the sampling.

        Returns:
            Dictionary mapping anchor bytes to their normalized sampling probability.
        """
        raw_anchors = []
        for ev in evidence_list:
            # Simplified anchor extraction: remove whitespace and slide a window
            clean_text = re.sub(r"\s+", "", ev)
            n = 8
            for i in range(0, len(clean_text) - n, 4):
                raw_anchors.append(clean_text[i : i + n].encode("utf-8"))

        if not raw_anchors:
            return {}

        # Count frequencies: higher frequency leads to lower weight (1/freq)
        counts = defaultdict(int)
        for a in raw_anchors:
            counts[a] += 1

        # Calculate importance weight: combines length and rarity
        weights = {a: (len(a) / (counts[a] ** 2)) for a, count in counts.items()}
        total_w = sum(weights.values())
        return {a: w / total_w for a, w in weights.items()}

    def get_roi(self, evidence_list: List[str], k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs Monte Carlo sampling to identify Regions of Interest (ROI) within the file.

        Args:
            evidence_list: List of reference strings to search for.
            k: Number of top candidate regions to return.

        Returns:
            List of dictionaries containing matched content, scores, and metadata. Output format:
                [
                  {
                    "content": "嵌入维度**：从1536增加到4096。\n- **更大的Patch Size**：使用16x16的patch size。\n- **RoPE位置编码**：采用旋转位置编码（Rotary Positional Embeddings），增强了模型对不同分辨率和长宽比的适应性。",
                    "score": 70.3,
                    "meta": {
                      "range": [
                        5317,
                        5668
                      ],
                      "hit_count": 1
                    }
                  },
                  ... ,
                ]
        """
        anchor_probs = self._get_weighted_anchors(evidence_list)
        if not anchor_probs:
            return []

        anchors = list(anchor_probs.keys())
        probs = list(anchor_probs.values())

        # Sample anchors based on calculated probabilities
        sampled_anchors = np.random.choice(
            anchors, size=min(len(anchors), self.sample_size), p=probs, replace=False
        )
        hit_map = defaultdict(list)

        def scan_anchor(anchor: bytes) -> List[Tuple[int, int]]:
            pos_list = []
            # Monte Carlo sampling: start searching from a random file offset
            search_start = random.randint(0, max(0, self.file_size - 1024))
            idx = self.mm.find(anchor, search_start)

            if idx != -1:
                # Key: identify the semantic boundaries surrounding the match point
                p_start = self._find_boundary(idx, direction="back")
                p_end = self._find_boundary(idx, direction="forward")
                pos_list.append((p_start, p_end))
            return pos_list

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(scan_anchor, a) for a in sampled_anchors]
            for f in as_completed(futures):
                for bounds in f.result():
                    # 'bounds' is a (start, end) tuple used as a dictionary key
                    hit_map[bounds].append(bounds)

        candidates = []
        for (p_start, p_end), hits in hit_map.items():
            # Boundary check
            p_start = max(0, p_start)
            p_end = min(self.file_size, p_end)

            if p_start >= p_end:
                continue

            p_text = self.mm[p_start:p_end].decode("utf-8", errors="ignore").strip()
            if not p_text:
                continue

            # Calculate similarity score using fuzzy matching against evidence snippets
            max_fuzz_score = max(
                [fuzz.partial_ratio(ev, p_text) for ev in evidence_list]
            )

            candidates.append(
                {
                    "content": p_text,
                    "score": round(float(max_fuzz_score), 2),
                    "meta": {"range": [p_start, p_end], "hit_count": len(hits)},
                }
            )

        # Rank candidates by relevance score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]

    def _find_boundary(self, pos: int, direction: str = "back") -> int:
        """
        Greedily search for the nearest semantic boundary in a specified direction.

        Args:
            pos (int): Starting position for the search.
            direction (str): Search direction, either 'back' or 'forward'.

        Returns:
            int: The detected boundary offset.
        """
        step = 1024
        # Pattern: Paragraph break (\n\n), Sentence break (punc + \n), or Line break (\n)
        pattern = re.compile(r"\r?\n\s*\r?\n|[\u3002\uff01\uff1f\.!\?]\s*\r?\n|\r?\n")

        for offset in range(0, self.max_scan, step):
            if direction == "back":
                start = max(0, pos - offset - step)
                end = pos - offset
            else:
                start = pos + offset
                end = min(self.file_size, pos + offset + step)

            if start >= end:
                break

            chunk = self.mm[start:end].decode("utf-8", errors="ignore")

            if direction == "back":
                matches = list(pattern.finditer(chunk))
                if matches:
                    return start + matches[-1].end()
                if start == 0:
                    return 0
            else:
                match = pattern.search(chunk)
                if match:
                    return start + match.start()
                if end == self.file_size:
                    return self.file_size

        return 0 if direction == "back" else self.file_size

    def close(self):
        """Releases memory map and file handles."""
        if hasattr(self, "mm"):
            self.mm.close()
        if hasattr(self, "f"):
            self.f.close()


if __name__ == "__main__":

    import json

    file_path: str = "/path/to/DINOv3_zh/report.md"

    with MonteCarloEvidenceSampling(file_path=file_path) as processor:

        results = processor.get_roi(
            [
                "**表 2: DINOv2 与 DINOv3 教师模型架构对比**\n",
                "DINOv3将模型规模扩展到了70亿（7B）参数，远超DINOv2的11亿参数。关键架构更新包括：\n",
                "- **3D理解** (表 13): 将VGGT框架中的DINOv2替换为DINOv3 ViT-L，在相机姿态估计、多视图重建等任务上均取得了一致的性能提升。\n",
            ]
        )

        # Outputs:
        # [
        #   {
        #     "content": "嵌入维度**：从1536增加到4096。\n- **更大的Patch Size**：使用16x16的patch size。\n- **RoPE位置编码**：采用旋转位置编码（Rotary Positional Embeddings），增强了模型对不同分辨率和长宽比的适应性。\n\n**表 2: DINOv2 与 DINOv3 教师模型架构对比**\n\n| 特性 | DINOv2 (ViT-giant) | DINOv3 (ViT-7B) |",
        #     "score": 70.3,
        #     "meta": {
        #       "range": [
        #         5317,
        #         5668
        #       ],
        #       "hit_count": 1
        #     }
        #   },
        #   ... ,
        # ]

        print(json.dumps(results, ensure_ascii=False, indent=2))
