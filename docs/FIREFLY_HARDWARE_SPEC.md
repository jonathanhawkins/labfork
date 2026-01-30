# Firefly Network Hardware Specification

> Version 0.1.0 - Prototype Phase
> Last Updated: January 2026

## Overview

The Firefly is a solar-powered, mesh-networked LED light designed for off-grid communities. Each unit provides illumination, phone charging capability, and connectivity to neighboring units within 1km range.

## Design Goals

| Requirement | Target | Priority |
|------------|--------|----------|
| Total BOM cost | < $25 USD (at 1k volume) | Critical |
| Operating temperature | -10C to +50C | Critical |
| Water resistance | IP65 minimum | Critical |
| Runtime on full charge | 12+ hours (medium brightness) | Critical |
| Mesh communication range | 500m+ (line of sight) | High |
| Solar charge time | < 8 hours (full sun) | High |
| Battery cycle life | 2000+ cycles | High |
| Weight | < 500g | Medium |
| Expected lifespan | 5+ years | Medium |

## Bill of Materials

### Core Components

| Component | Part Number | Unit Cost | Qty | Total | Notes |
|-----------|-------------|-----------|-----|-------|-------|
| ESP32-C6-WROOM-1 | ESP32-C6-WROOM-1-N8 | $3.50 | 1 | $3.50 | Thread + WiFi + BLE |
| Solar Panel 5W | Generic mono 5V | $4.00 | 1 | $4.00 | 170x220mm |
| LiFePO4 Cell 3.2V 6Ah | IFR32650 | $6.00 | 1 | $6.00 | 2000+ cycles |
| LED COB 10W 3000K | Generic | $2.50 | 1 | $2.50 | CRI 80+, 1000lm |
| **Subtotal Core** | | | | **$16.00** | |

### Power Management

| Component | Part Number | Unit Cost | Qty | Total | Notes |
|-----------|-------------|-----------|-----|-------|-------|
| MPPT IC | CN3722 | $0.50 | 1 | $0.50 | Solar MPPT controller |
| Battery Protection IC | S-8261ABJMD-G3JT2G | $0.30 | 1 | $0.30 | OVP/UVP/OCP |
| Buck Converter 3.3V | MP2315 | $0.40 | 1 | $0.40 | For ESP32 |
| LED Driver | PT4115 | $0.30 | 1 | $0.30 | PWM dimmable |
| USB-C Connector | Generic | $0.20 | 1 | $0.20 | Phone charging |
| USB PD IC | CH224K | $0.50 | 1 | $0.50 | 5V/9V negotiation |
| **Subtotal Power** | | | | **$2.20** | |

### Passive Components

| Component | Value | Unit Cost | Qty | Total | Notes |
|-----------|-------|-----------|-----|-------|-------|
| Capacitors (various) | 100nF-100uF | $0.02 | 20 | $0.40 | MLCC + electrolytic |
| Resistors (various) | 1R-1M | $0.01 | 30 | $0.30 | 0603 |
| Inductors | 4.7uH-22uH | $0.10 | 3 | $0.30 | Shielded |
| Schottky Diodes | SS34 | $0.05 | 5 | $0.25 | Reverse protection |
| MOSFETs | AO3400 | $0.05 | 4 | $0.20 | Switching |
| **Subtotal Passives** | | | | **$1.45** | |

### Mechanical

| Component | Description | Unit Cost | Qty | Total | Notes |
|-----------|-------------|-----------|-----|-------|-------|
| PCB | 2-layer, 100x80mm | $0.80 | 1 | $0.80 | HASL, FR4 |
| Enclosure | ABS IP65 box | $2.50 | 1 | $2.50 | 150x100x50mm |
| Lens/Diffuser | Polycarbonate | $0.40 | 1 | $0.40 | 120 beam |
| Mounting bracket | Aluminum | $0.30 | 1 | $0.30 | Wall/pole mount |
| Cable glands | PG7 | $0.10 | 2 | $0.20 | IP68 |
| Screws/fasteners | Stainless | $0.15 | 1 | $0.15 | M3, M4 assorted |
| **Subtotal Mechanical** | | | | **$4.35** | |

### Misc

| Component | Description | Unit Cost | Qty | Total | Notes |
|-----------|-------------|-----------|-----|-------|-------|
| Wire | 22AWG silicone | $0.20 | 1 | $0.20 | Various colors |
| Connectors | JST-PH | $0.30 | 1 | $0.30 | Battery, solar |
| Thermal paste | Generic | $0.10 | 1 | $0.10 | LED heatsinking |
| Conformal coating | Silicone spray | $0.20 | 1 | $0.20 | PCB protection |
| Label | Weatherproof | $0.05 | 1 | $0.05 | QR code for setup |
| **Subtotal Misc** | | | | **$0.85** | |

### BOM Summary

| Category | Cost |
|----------|------|
| Core Components | $16.00 |
| Power Management | $2.20 |
| Passive Components | $1.45 |
| Mechanical | $4.35 |
| Misc | $0.85 |
| **TOTAL** | **$24.85** |

*Note: Prices based on 1,000 unit volume from LCSC/Alibaba. Lower costs achievable at 10k+ volume.*

## Block Diagram

```
                    ┌─────────────┐
                    │  Solar 5W   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    MPPT     │
                    │   CN3722    │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
     ┌──────▼──────┐ ┌─────▼─────┐ ┌─────▼─────┐
     │   Battery   │ │  3.3V DC  │ │  5V USB   │
     │  LiFePO4    │ │   Buck    │ │   Out     │
     │   6Ah       │ └─────┬─────┘ └───────────┘
     └─────────────┘       │
                     ┌─────▼─────┐
                     │  ESP32-C6 │
                     │  Thread   │
                     │  WiFi/BLE │
                     └─────┬─────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
     ┌──────▼──────┐ ┌─────▼─────┐ ┌─────▼─────┐
     │  LED Driver │ │  Sensors  │ │   Mesh    │
     │   PT4115    │ │ Temp/Light│ │  Antenna  │
     └──────┬──────┘ └───────────┘ └───────────┘
            │
     ┌──────▼──────┐
     │  LED 10W    │
     │  1000lm     │
     └─────────────┘
```

## PCB Layout Guidelines

### Layer Stack
- Layer 1: Signal + Components
- Layer 2: Ground plane

### Critical Traces
- Solar input: 2mm minimum width
- Battery: 3mm minimum width
- LED power: 2mm minimum width
- ESP32 signals: 0.2mm minimum

### Thermal Management
- LED pad: Large copper pour, thermal vias to back
- Battery area: Keep away from heat sources
- MPPT IC: Copper pour for heatsinking

### EMI Considerations
- Keep Thread antenna area clear of copper
- Separate analog (solar/battery) from digital
- Ground plane under ESP32

## Enclosure Design

### Dimensions
- External: 150 x 100 x 50 mm
- Wall thickness: 2mm
- Material: ABS (UV stabilized)

### Features
- Solar panel mounting on top surface
- LED lens/diffuser window
- USB-C port with rubber flap
- Button for mode selection
- Status LED window
- Wall/pole mounting holes
- Cable gland for external wiring

### Sealing
- IP65 rating required
- Silicone gasket between lid and body
- Cable glands for all penetrations
- Conformal coating on PCB

## Assembly Instructions

### Prerequisites
- Soldering station (temperature controlled)
- Hot air rework station
- Multimeter
- Oscilloscope (recommended)
- Thermal camera (recommended)

### Assembly Order
1. SMD components (reflow or hand solder)
2. Through-hole components
3. Connectors
4. LED module (with thermal compound)
5. Functional test (before enclosure)
6. Conformal coating
7. Final assembly in enclosure
8. Sealing and final test

### Quality Checkpoints
- [ ] Visual inspection after SMD
- [ ] Power rails voltage check
- [ ] ESP32 programming verification
- [ ] LED brightness test (all levels)
- [ ] Mesh connectivity test
- [ ] Thermal test (1 hour full brightness)
- [ ] IP65 water spray test

## Test Procedures

### Power System Test
1. Connect to bench supply at 5V
2. Verify 3.3V rail: 3.3V +/- 5%
3. Verify LED driver output at each PWM level
4. Measure efficiency at full brightness

### Solar Charging Test
1. Connect solar simulator or real panel
2. Verify MPPT tracking (measure input power vs battery power)
3. Full charge cycle (measure time and efficiency)

### Battery Test
1. Full discharge at rated load
2. Measure capacity (should be >90% rated)
3. Verify cutoff voltages (high and low)

### Mesh Connectivity Test
1. Program two units
2. Verify Thread mesh formation
3. Range test at 100m, 300m, 500m, 1000m
4. Verify message delivery reliability

### Environmental Test
1. Temperature cycling: -10C to +50C
2. IP65 water spray test
3. UV exposure (accelerated aging)
4. Vibration test (transport simulation)

## Safety Considerations

### Electrical
- LiFePO4 chemistry chosen for thermal stability
- Battery protection IC prevents overcharge/discharge
- Fused solar input
- Current-limited LED driver

### Mechanical
- No sharp edges
- Secure battery retention
- Strain relief on all cables

### Thermal
- LED thermal management critical
- Auto-dimming at high temperature
- Battery thermal monitoring

## Certifications Needed (for Production)

- CE marking (Europe)
- FCC Part 15 (US)
- RoHS compliance
- UN38.3 battery transport
- IP65 verification

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | Jan 2026 | Initial draft |

## Contributors

- Add your name here when you contribute!
