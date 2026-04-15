# IMPROVEMENT BREAKDOWN

## Best Model (LR=0.001) vs Others

### vs. First Experiment (LR=0.0001)
- LR=0.0001: 15% success
- LR=0.001:  60% success
- **Improvement: +45 percentage points (4x better)**

### vs. Middle Experiment (LR=0.0005)  
- LR=0.0005: 30% success (final)
- LR=0.001:  60% success
- **Improvement: +30 percentage points (2x better)**

### Performance Comparison Table
| LR | Success Rate | vs Best | Improvement |
|-----|-------------|---------|-------------|
| 0.001 | 60% | baseline | — |
| 0.0005 | 30% | -30pp | 50% worse |
| 0.0001 | 15% | -45pp | 75% worse |

## Key Insight
The lr=0.001 model is:
- **4x more successful than the conservative lr=0.0001**
- **2x more successful than the moderate lr=0.0005**
- Represents the optimal learning rate for this task

