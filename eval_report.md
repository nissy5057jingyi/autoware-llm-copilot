# DriveTDPA: VL-DPO vs Numeric — Full Metric Evaluation

## Per-Scene Results

| Scene | Goal | Num ADE↓ | VL ADE↓ | Num Goal↑ | VL Goal↑ | Num Cons↑ | VL Cons↑ | Agree |
|-------|------|---------|---------|----------|----------|----------|----------|-------|
| Clear road, straight ahea | FORWARD | 3.792 | 3.792 | 1.000 | 1.000 | 1.000 | 1.000 | ✓ |
| Static obstacle 15m ahead | FORWARD | 0.000 | 3.583 | 1.000 | 0.964 | 0.833 | 1.000 | ✗ |
| Slow vehicle ahead, left  | FORWARD | 0.000 | 0.000 | 0.368 | 0.368 | 0.667 | 0.667 | ✓ |
| Intersection, turn right | RIGHT | 0.000 | 0.000 | 0.990 | 0.990 | 0.500 | 0.500 | ✓ |
| Rainy road, cautious forw | FORWARD | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✓ |
| Night driving, maintain s | FORWARD | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✓ |
| Pedestrian crossing, emer | FORWARD | 0.000 | 3.333 | 0.982 | 1.000 | 1.000 | 0.833 | ✗ |
| Highway, high speed forwa | FORWARD | 0.000 | 6.417 | 1.000 | 1.000 | 1.000 | 1.000 | ✗ |
| Intersection, turn left | LEFT | 0.000 | 0.000 | 0.990 | 0.990 | 0.500 | 0.500 | ✓ |
| Vehicle cutting in from r | FORWARD | 0.000 | 0.000 | 1.000 | 1.000 | 0.833 | 0.833 | ✓ |

## Average Metrics

| Metric | Numeric | VL-DPO | Delta | Better |
|--------|---------|--------|-------|--------|
| ADE↓ | 0.3792 | 1.7125 | -1.3333 | Numeric |
| FDE↓ | 0.9000 | 3.2700 | -2.3700 | Numeric |
| Goal↑ | 0.9330 | 0.9312 | -0.0018 | Numeric |
| Consistency↑ | 0.8333 | 0.8333 | -0.0000 | Tie |
| Unsup↓ | 0.0667 | 0.0667 | -0.0000 | Tie |
| Lat↓ | 1.3600 | 1.3600 | -0.0000 | Tie |

## Key Statistics

- Agreement rate: 70% (7/10 scenes)
- Training signal: VL-DPO generates 3.0x more preference pairs
- Total scenes: 10 (Day/Rain/Night conditions)