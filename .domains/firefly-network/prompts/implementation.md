# Firefly Network Implementation Prompt

You are implementing components for The Firefly Network hardware and firmware.

## Design Constraints

- **Total BOM**: Must stay under $25 USD
- **Environment**: IP65 rating, -10C to 50C operating range
- **Power**: 5W solar input, 12+ hour runtime at medium brightness
- **Mesh**: 500m+ range between nodes, self-healing network
- **Repairability**: Must be fixable with basic tools

## Hardware Platform

### Core Components
- **MCU**: ESP32-C6 (Thread + WiFi + BLE)
- **Solar**: 5W monocrystalline panel
- **Battery**: LiFePO4 3.2V 6Ah cell
- **LED**: High-CRI COB array, 1000lm max
- **Charging**: Custom MPPT circuit

### Communication Stack
- **Primary**: Thread mesh (IEEE 802.15.4)
- **Backup**: BLE for configuration
- **Optional**: LoRa for extended range

## Firmware Architecture

```
firefly-firmware/
├── src/
│   ├── main.c              # Entry point, task scheduler
│   ├── mesh/               # Thread mesh implementation
│   │   ├── mesh_manager.c
│   │   ├── routing.c
│   │   └── neighbor_table.c
│   ├── power/              # Power management
│   │   ├── mppt.c          # MPPT algorithm
│   │   ├── battery.c       # BMS interface
│   │   └── sleep.c         # Low-power modes
│   ├── light/              # LED control
│   │   ├── pwm_driver.c
│   │   ├── dimming.c
│   │   └── scheduler.c     # Auto on/off
│   ├── swarm/              # Swarm intelligence
│   │   ├── consensus.c     # Distributed decisions
│   │   ├── energy_share.c  # Power sharing
│   │   └── coverage.c      # Network optimization
│   └── config/             # Configuration
│       ├── nvs_storage.c
│       └── ble_config.c
├── include/
└── test/
```

## Implementation Guidelines

### Power Efficiency
- Use ESP32-C6 light sleep between mesh operations
- Target <50mA average current draw
- Implement aggressive duty cycling

### Mesh Reliability
- Implement 3-hop redundancy
- Automatic rerouting on node failure
- Periodic heartbeat with exponential backoff

### Safety
- Over-temperature protection
- Battery over-discharge protection
- Graceful degradation under low power

## Code Style

- Use ESP-IDF framework
- Follow ESP-IDF coding standards
- Comprehensive error handling
- Document all public APIs
- Write unit tests for critical paths

## Testing Requirements

- Unit tests for MPPT algorithm
- Integration tests for mesh formation
- Power consumption measurements
- Environmental stress testing
