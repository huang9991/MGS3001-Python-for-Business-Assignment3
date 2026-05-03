# Data Description: Reddit Mental Health Posts

## Data Source
- **Platform**: Reddit
- **Archive**: [Arctic Shift](https://arctic-shift.photon-reddit.com/download-tool)
- **Subreddits**: r/mentalhealth, r/anxiety, r/depression, r/psychology
- **Rationale**: Reddit API restrictions (post-June 2023) prevent bulk historical data collection; Arctic Shift provides complete subreddit archives with consistent field coverage and is used in recent academic research (e.g., del Rio-Chanona et al., 2024; Shan & Qiu, 2025).

## Collection Method
1.  **Download**: JSONL files from Arctic Shift download tool.
2.  **Field Selection**: 8 core fields retained from 91–126 available fields: `id`, `subreddit`, `title`, `selftext`, `created_utc`, `score`, `ups`, `num_comments`.
3.  **Cleaning**:
    -   Timestamp conversion: Unix `created_utc` → datetime `date`.
    -   Text merging: `title` + `selftext` combined into single `text` field.
    -   Filtering: Removed `[deleted]`/`[removed]` posts and texts < 20 characters.
    -   Filtered out 134,051 records (23.7%).
4.  **Derived Fields**: `type` (post/comment), `period` (before/after ChatGPT release), `days_from_cutoff1`, `days_from_cutoff2`.

## Time Windows
| Cutoff | Date | Event |
|---|---|---|
| Cutoff 1 | 2022-11-30 | ChatGPT public release |
| Cutoff 2 | 2024-09-30 | Widespread ChatGPT adoption (Chatterji et al., 2025) |
| Data start | 2021-09-01 | |
| Data end | 2026-04-28 | |

## Dataset Overview
| Metric | Value |
|---|---|
| Records (raw) | 565,354 |
| Records (clean) | 431,303 |
| Variables | 11 |
| File size | ~510.8 MB |

## Period Distribution
| Period | N | % |
|---|---|---|
| Before ChatGPT (Cutoff 1) | 83,861 | 19.4% |
| After ChatGPT (Cutoff 1) | 347,442 | 80.6% |

## Descriptive Statistics (Key Numeric Variables)
| Statistic | score | num_comments | days_from_cutoff1 | text_length |
|---|---|---|---|---|
| Mean | 3.88 | 3.15 | 527.73 | 1,132 |
| SD | 21.54 | 11.09 | 480.42 | 1,063 |
| Min | 0 | 0 | −455 | 20 |
| 25% | 1 | 0 | 166 | 458 |
| 50% | 1 | 1 | 582 | 855 |
| 75% | 2 | 3 | 941 | 1,483 |
| Max | 2,344 | 1,040 | 1,245 | 40,094 |

**Key observations**:
- Engagement metrics (`score`, `num_comments`) are heavily right-skewed (median = 1, mean ≈ 3–4).
- `days_from_cutoff1` ranges from −455 to +1,245, providing sufficient bandwidth for RDD analysis.
- Mean text length = 1,132 characters (SD = 1,063), suitable for NLP feature extraction.

## Output Variables
| # | Variable | Type | Description |
|---|---|---|---|
| 1 | `id` | string | Unique post identifier |
| 2 | `subreddit` | string | Source subreddit (e.g., "mentalhealth") |
| 3 | `type` | string | Content type ("post") |
| 4 | `date` | datetime | Post timestamp (UTC) |
| 5 | `text` | string | Combined title + selftext |
| 6 | `score` | integer | Net upvotes (ups − downs) |
| 7 | `ups` | integer | Raw upvote count |
| 8 | `num_comments` | integer | Number of comments |
| 9 | `period` | string | "before" or "after" Cutoff 1 |
| 10 | `days_from_cutoff1` | integer | Days relative to ChatGPT release |
| 11 | `days_from_cutoff2` | integer | Days relative to widespread adoption |

> **Note**: NLP-derived features (e.g., `psychoedu_score`, `psych_term_density`, `support_score`) are computed downstream by `FeatureExtractor` and not included in this cleaned dataset.

## Limitations
- **Class imbalance**: The "after" period contains ~4× more posts than "before," which may affect pre-post comparisons.
- **Missing IDs**: Row indices are non-consecutive, indicating records removed during filtering.
- **Subreddit coverage**: This file covers r/mentalhealth posts only; other subreddits and comments require separate processing.
