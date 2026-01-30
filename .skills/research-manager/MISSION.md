# Research Manager Mission

## Epic: The Firefly Network - Solar-Powered Mesh Lighting

**North Star Goal:** Build ultra-low-cost (<$25) solar-powered lights that form autonomous mesh networks, bringing illumination, power, and connectivity to 1 billion people without electricity.

**Success Criteria:**
- [ ] Thread mesh networking on ESP32-C6 with 500m+ range
- [ ] MPPT solar charging achieving >90% efficiency
- [ ] 12+ hour runtime on full charge at medium brightness
- [ ] Total BOM cost under $25
- [ ] IP65 weather resistance
- [ ] Swarm coordination for energy sharing

---

## Current Stories (Milestones)

### S1: Mesh Protocol Implementation (ACTIVE)
**Goal:** Implement reliable mesh networking for ESP32-C6.

**Acceptance Criteria:**
- Thread protocol running on ESP32-C6
- 500m+ range between nodes
- Self-healing network topology
- Energy-efficient routing

**Research Areas:**
- Thread vs Zigbee vs LoRa comparison
- Energy-aware routing protocols
- Network self-healing algorithms

---

### S2: Solar Power Optimization
**Goal:** Maximize energy harvesting from small solar panels.

**Acceptance Criteria:**
- MPPT algorithm achieving >90% efficiency
- Works with 5-10W panels
- Handles partial shading gracefully
- Adaptive to weather conditions

**Research Areas:**
- Perturb & Observe vs Incremental Conductance
- Fuzzy logic MPPT
- Multi-panel optimization

---

### S3: LED & Light Quality
**Goal:** Design efficient, human-friendly lighting.

**Acceptance Criteria:**
- 400+ lumens at 3W
- CRI > 90 for color accuracy
- 3000K warm white
- Even light distribution

**Research Areas:**
- High-efficiency LED selection
- Thermal management
- Optics design

---

### S4: Swarm Intelligence
**Goal:** Implement distributed coordination without central server.

**Acceptance Criteria:**
- Nodes synchronize brightness patterns
- Energy sharing during emergencies
- Graceful degradation with node failures
- No single point of failure

**Research Areas:**
- Firefly synchronization algorithms
- Distributed consensus
- Energy-aware collective behavior

---

### S5: Battery Management
**Goal:** Maximize battery life and safety.

**Acceptance Criteria:**
- 2000+ charge cycles
- Works from -20C to 60C
- Safe charging from solar
- Accurate state of charge estimation

**Research Areas:**
- LiFePO4 management
- BMS circuit design
- Temperature compensation

---

## Research Backlog

Promising techniques to investigate:

1. **Thread Border Router** (OpenThread)
   - Reference implementation for ESP32
   - Integration with IPv6

2. **Bio-Inspired Synchronization** (Firefly Algorithm)
   - Pulse-coupled oscillators
   - Phase synchronization

3. **Perturb & Observe MPPT**
   - Simple, effective for small panels
   - Well-documented implementations

4. **Energy-Aware Routing (LEACH)**
   - Cluster-based routing
   - Load balancing

5. **LiFePO4 BMS Designs**
   - Open source BMS projects
   - Protection circuits

---

## Manager Operating Instructions

1. **Check Progress:** Start each session by reviewing this file and TaskList
2. **Pick Next Work:** Choose highest-priority unblocked story/task
3. **Research First:** Before implementing, search arxiv for relevant papers
4. **Spawn Specialists:** Use Ollama for research, Codex for deep analysis
5. **Track Everything:** Create tasks, update status, document findings
6. **Focus on Prototype:** We're building a working prototype THIS MONTH
7. **Cost Constraint:** Every decision must consider the $25 BOM target

**Current Priority:** S1 (Mesh Protocol Implementation)

---

## Hardware Targets

| Component | Target | Notes |
|-----------|--------|-------|
| MCU | ESP32-C6 | Thread + WiFi + BLE |
| Solar Panel | 5-10W | Monocrystalline |
| Battery | LiFePO4 3.2V 6Ah | Long cycle life |
| LED | 1000lm/3W | High CRI |
| Total BOM | <$25 | At scale |

---

## Key Constraints

- **Humanitarian Focus:** Prioritize reliability over features
- **Repair Friendly:** Must be fixable with basic tools
- **Harsh Conditions:** Rain, dust, extreme temperatures
- **No Internet Required:** Must work fully offline
- **Energy Autonomous:** Must be self-sufficient on solar

---

## Version History

- 2026-01-29: Pivoted from Voice Clone to Firefly Network
- Initial focus: Mesh networking + solar optimization
- Target: Working prototype by end of month
