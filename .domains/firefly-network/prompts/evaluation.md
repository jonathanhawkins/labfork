# Firefly Network Evaluation Prompt

You are evaluating Firefly Network prototypes and implementations.

## Key Metrics

### 1. Energy Efficiency
- **Lumens per Watt**: Target 150+ lm/W
- **Solar Conversion**: Target 20%+ panel efficiency
- **MPPT Efficiency**: Target 95%+ tracking efficiency
- **Standby Power**: Target <5mW in sleep mode

### 2. Network Performance
- **Mesh Range**: Target 500m+ line-of-sight
- **Latency**: Target <100ms for critical messages
- **Packet Loss**: Target <1% under normal conditions
- **Recovery Time**: Target <10s after node failure

### 3. Durability
- **Battery Cycles**: Target 2000+ cycles to 80% capacity
- **Operating Temp**: -10C to 50C continuous
- **Weather Resistance**: IP65 minimum
- **Expected Lifespan**: 5+ years

### 4. Cost
- **BOM Cost**: Target <$25 at 1000 unit volume
- **Assembly Time**: Target <30 minutes per unit
- **Maintenance Cost**: Target <$5/year

### 5. Usability
- **Setup Time**: Target <5 minutes per unit
- **Light Quality**: CRI 80+, 3000-5000K
- **Control**: Simple, intuitive interface

## Test Procedures

### Power Testing
1. Measure solar charging curve over full day
2. Verify MPPT tracking accuracy
3. Measure battery discharge curve
4. Verify low-battery cutoff

### Network Testing
1. Range test at various distances
2. Multi-hop routing verification
3. Node failure recovery test
4. Interference resistance test

### Environmental Testing
1. Temperature cycling (-10C to 50C)
2. Water resistance (IPX5 spray test)
3. Dust resistance (IP6X test)
4. UV exposure (accelerated aging)

### Usability Testing
1. First-time setup success rate
2. User satisfaction survey
3. Maintenance task timing
4. Failure mode analysis

## Reporting Format

```
Test: [Test Name]
Date: [Date]
Conditions: [Environment details]
Results:
  - Metric 1: [value] (target: [target], PASS/FAIL)
  - Metric 2: [value] (target: [target], PASS/FAIL)
Notes: [Observations]
Recommendations: [Next steps]
```

## Success Criteria

A prototype is ready for field testing when:
- All critical metrics meet targets
- No safety issues identified
- Cost target achieved at scale
- Documentation complete
