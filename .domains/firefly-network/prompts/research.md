# Firefly Network Research Prompt

You are researching technologies for The Firefly Network - an open-source project to bring solar-powered mesh lighting to communities without electricity.

## Research Focus Areas

### 1. Solar Energy Optimization
- MPPT (Maximum Power Point Tracking) algorithms for small panels (<20W)
- Solar cell efficiency in varying conditions (angle, temperature, partial shading)
- Energy harvesting from ambient light sources
- Solar panel degradation and longevity

### 2. Mesh Networking
- Thread protocol implementation on ESP32-C6
- LoRa for long-range mesh (1km+)
- Energy-efficient routing protocols
- Network self-healing and resilience
- Latency requirements for real-time coordination

### 3. Swarm Intelligence
- Distributed consensus without central coordinator
- Energy sharing algorithms between nodes
- Collective decision-making for network optimization
- Emergent behavior patterns for coverage optimization

### 4. Battery & Power Management
- LiFePO4 vs Li-ion for longevity and safety
- BMS (Battery Management System) design
- Low-power sleep modes for microcontrollers
- Power budgeting for 12+ hour runtime

### 5. Light Quality & Human Factors
- Optimal color temperature for various tasks (reading, cooking)
- Circadian rhythm considerations
- Light distribution patterns
- Dimming and control interfaces

### 6. Hardware Design
- PCB design for harsh environments
- IP65+ enclosure design
- Component sourcing for low cost
- Design for manufacturability at scale
- Repairability with basic tools

## Evaluation Criteria

When evaluating papers, prioritize:
1. **Cost impact**: Will this reduce BOM cost?
2. **Reliability**: Will this work in harsh conditions?
3. **Simplicity**: Can this be implemented with basic components?
4. **Proven**: Has this been validated in real-world conditions?

## Output Format

For each paper, provide:
- Key techniques applicable to Firefly Network
- Estimated implementation complexity (1-5)
- Potential cost/performance impact
- Specific component or algorithm recommendations
