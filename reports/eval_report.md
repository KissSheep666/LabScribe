# LabScribe Evaluation Report

Generated: `2026-11-17T05:17:09+00:00`

## Dataset

- scenarios: **100**
- ground-truth events: **414**
- occluded ground-truth events: **23**
- backends: `marker` / `iou` / `template`

## Event detection

| type | TP | FP | FN | precision | recall | F1 | mean latency (s) | p95 latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **overall** | 414 | 0 | 0 | 100.0% | 100.0% | 100.0% | 0.408 | 1.47 |
| anomaly | 14 | 0 | 0 | 100.0% | 100.0% | 100.0% | 0.969 | 1.55 |
| reading_change | 128 | 0 | 0 | 100.0% | 100.0% | 100.0% | 0.921 | 1.58 |
| session_end | 136 | 0 | 0 | 100.0% | 100.0% | 100.0% | 0.017 | 0.08 |
| session_start | 136 | 0 | 0 | 100.0% | 100.0% | 100.0% | 0.26 | 0.39 |

## OCR / readings

- raw OCR sample accuracy: **98.23%** (3779 samples)
- visible-frame reading accuracy: **100.0%** (3709 frames)
- event reading accuracy: **100.0%** (142 readings compared)

## Occlusion

- occluded GT events: **23**
- matched: **23**, missed: **0**
- occlusion recall: **100.0%**

## Runtime

- frames processed: **11809**
- mean frame processing: **1.259 ms**
- pending (low-confidence) candidates: **0**
- eval wall clock: **173.22 s**
