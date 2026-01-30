# Firefly Network Software Architecture

> Version 0.1.0 - Prototype Phase
> Last Updated: January 2026

## Overview

The Firefly firmware runs on ESP32-C6, implementing mesh networking, solar power management, LED control, and swarm intelligence algorithms. The system is designed for extreme reliability, low power consumption, and zero-configuration deployment.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Light Control │  Swarm Logic    │    User Interface           │
│   - Dimming     │  - Consensus    │    - BLE Configuration      │
│   - Scheduling  │  - Energy Share │    - Status Reporting       │
│   - Motion Det  │  - Coverage Opt │    - OTA Updates            │
├─────────────────┴─────────────────┴─────────────────────────────┤
│                       Mesh Network Layer                         │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Thread Stack  │   Routing       │    Message Queue            │
│   - IPv6        │   - Multi-hop   │    - Priority Queue         │
│   - CoAP        │   - Self-heal   │    - Retry Logic            │
│   - UDP         │   - Load Balance│    - Compression            │
├─────────────────┴─────────────────┴─────────────────────────────┤
│                        Hardware Layer                            │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Power Mgmt    │   Sensors       │    Peripherals              │
│   - MPPT        │   - Temperature │    - LED PWM                │
│   - Battery BMS │   - Light       │    - GPIO                   │
│   - Sleep Modes │   - Voltage/Cur │    - USB                    │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## Directory Structure

```
firefly-firmware/
├── CMakeLists.txt
├── sdkconfig.defaults
├── partitions.csv
├── main/
│   ├── CMakeLists.txt
│   ├── main.c                    # Entry point
│   └── Kconfig.projbuild         # Project config
├── components/
│   ├── mesh/
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   └── mesh_manager.h
│   │   ├── mesh_manager.c        # Thread mesh management
│   │   ├── routing.c             # Multi-hop routing
│   │   ├── neighbor_table.c      # Neighbor discovery
│   │   └── mesh_messages.c       # Message serialization
│   ├── power/
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   └── power_manager.h
│   │   ├── mppt.c                # MPPT algorithm
│   │   ├── battery.c             # BMS interface
│   │   ├── sleep.c               # Power states
│   │   └── energy_budget.c       # Energy accounting
│   ├── light/
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   └── light_controller.h
│   │   ├── pwm_driver.c          # LED PWM control
│   │   ├── dimming.c             # Dimming curves
│   │   └── scheduler.c           # Sunrise/sunset
│   ├── swarm/
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   └── swarm_logic.h
│   │   ├── consensus.c           # Distributed consensus
│   │   ├── energy_share.c        # Power sharing
│   │   └── coverage.c            # Network optimization
│   ├── config/
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   └── config_manager.h
│   │   ├── nvs_storage.c         # Persistent storage
│   │   └── ble_config.c          # BLE provisioning
│   └── sensors/
│       ├── CMakeLists.txt
│       ├── include/
│       │   └── sensors.h
│       ├── temperature.c         # Temp monitoring
│       ├── light_sensor.c        # Ambient light
│       └── power_monitor.c       # V/I sensing
├── test/
│   ├── test_mppt.c
│   ├── test_mesh.c
│   ├── test_consensus.c
│   └── CMakeLists.txt
└── docs/
    ├── API.md
    └── PROTOCOLS.md
```

## Core Components

### 1. Power Management (`components/power/`)

#### MPPT Algorithm
```c
// Perturb and Observe MPPT with adaptive step size
typedef struct {
    float voltage;
    float current;
    float power;
    float step_size;
    mppt_state_t state;
} mppt_context_t;

void mppt_init(mppt_context_t *ctx);
void mppt_update(mppt_context_t *ctx, float v_solar, float i_solar);
float mppt_get_duty_cycle(mppt_context_t *ctx);
```

**Key Features:**
- Perturb & Observe algorithm with adaptive step
- Partial shading detection
- Temperature compensation
- Tracking efficiency >95%

#### Battery Management
```c
typedef struct {
    float voltage;
    float current;
    float soc;  // State of Charge (0-100%)
    float soh;  // State of Health (0-100%)
    float temperature;
    battery_state_t state;
} battery_context_t;

void battery_init(battery_context_t *ctx);
void battery_update(battery_context_t *ctx);
bool battery_can_discharge(battery_context_t *ctx);
bool battery_can_charge(battery_context_t *ctx);
```

**Key Features:**
- LiFePO4-specific charging profile
- Coulomb counting + OCV for SOC
- Temperature protection
- Cycle counting for SOH estimation

#### Sleep Management
```c
typedef enum {
    POWER_STATE_ACTIVE,      // Full operation
    POWER_STATE_LIGHT_SLEEP, // CPU sleeping, peripherals on
    POWER_STATE_DEEP_SLEEP,  // Minimal power, timer wake
    POWER_STATE_HIBERNATION, // Battery protection mode
} power_state_t;

void power_enter_state(power_state_t state);
void power_set_wakeup_timer(uint32_t seconds);
```

**Power Consumption Targets:**
- Active (mesh + LED): <500mA
- Light sleep (mesh only): <10mA
- Deep sleep: <100uA
- Hibernation: <10uA

### 2. Mesh Networking (`components/mesh/`)

#### Thread Mesh Manager
```c
typedef struct {
    otInstance *ot_instance;
    mesh_role_t role;  // Leader, Router, Child
    uint8_t network_key[16];
    uint16_t pan_id;
    uint8_t channel;
} mesh_context_t;

void mesh_init(mesh_context_t *ctx);
void mesh_start(mesh_context_t *ctx);
bool mesh_is_connected(mesh_context_t *ctx);
void mesh_send_message(mesh_context_t *ctx, mesh_msg_t *msg);
```

**Key Features:**
- OpenThread stack on ESP32-C6
- Automatic role negotiation
- Self-healing network
- Up to 250 nodes per network

#### Message Types
```c
typedef enum {
    MSG_HEARTBEAT = 0x01,      // Periodic keepalive
    MSG_ENERGY_STATUS = 0x02,  // Battery/solar status
    MSG_LIGHT_COMMAND = 0x03,  // Set brightness
    MSG_CONSENSUS_VOTE = 0x04, // Swarm voting
    MSG_EMERGENCY = 0x05,      // Priority message
    MSG_OTA_ANNOUNCE = 0x06,   // Firmware update
} msg_type_t;

typedef struct {
    msg_type_t type;
    uint16_t source_id;
    uint16_t dest_id;  // 0xFFFF for broadcast
    uint8_t ttl;
    uint8_t payload_len;
    uint8_t payload[64];
    uint32_t timestamp;
} mesh_msg_t;
```

### 3. Light Control (`components/light/`)

#### PWM Driver
```c
typedef struct {
    uint8_t brightness;    // 0-255
    uint16_t color_temp;   // 2700-6500K (if dual LED)
    bool is_on;
    dim_curve_t curve;     // Linear, Logarithmic, etc.
} light_context_t;

void light_init(light_context_t *ctx);
void light_set_brightness(light_context_t *ctx, uint8_t level);
void light_fade_to(light_context_t *ctx, uint8_t level, uint16_t duration_ms);
```

**Key Features:**
- Flicker-free PWM at 20kHz
- Logarithmic dimming for natural feel
- Smooth transitions
- Temperature-based derating

#### Scheduler
```c
typedef struct {
    uint8_t hour;
    uint8_t minute;
    uint8_t brightness;
} schedule_point_t;

void scheduler_init(void);
void scheduler_add_point(schedule_point_t *point);
void scheduler_set_location(float latitude, float longitude);  // For sunrise/sunset
```

### 4. Swarm Intelligence (`components/swarm/`)

#### Distributed Consensus
```c
// RAFT-inspired lightweight consensus
typedef struct {
    uint16_t node_id;
    consensus_state_t state;  // Follower, Candidate, Leader
    uint32_t term;
    uint16_t voted_for;
    uint16_t leader_id;
} consensus_context_t;

void consensus_init(consensus_context_t *ctx);
void consensus_propose(consensus_context_t *ctx, proposal_t *proposal);
void consensus_on_vote(consensus_context_t *ctx, vote_t *vote);
```

**Key Features:**
- Lightweight RAFT variant
- Leader election for coordination
- Proposal/voting for network decisions
- Timeout-based failover

#### Energy Sharing
```c
typedef struct {
    uint16_t node_id;
    float soc;           // State of Charge
    float solar_power;   // Current generation
    float load;          // Current consumption
    bool needs_help;     // Low battery flag
    bool can_help;       // Excess energy flag
} energy_status_t;

void energy_share_init(void);
void energy_share_broadcast_status(energy_status_t *status);
void energy_share_request_help(uint16_t target_node);
```

**Algorithm:**
1. Nodes broadcast energy status periodically
2. Low-battery nodes set `needs_help` flag
3. Nearby nodes with excess energy offer to dim
4. Network redistributes load to help struggling nodes

#### Coverage Optimization
```c
typedef struct {
    float x, y;          // Estimated position
    float brightness;    // Current output
    float coverage_radius;
} node_position_t;

void coverage_init(void);
void coverage_optimize(node_position_t *nodes, size_t count);
float coverage_calculate_total(node_position_t *nodes, size_t count);
```

**Algorithm:**
- Estimate relative positions from signal strength
- Identify coverage gaps
- Adjust brightness to maximize coverage efficiency

### 5. Configuration (`components/config/`)

#### BLE Provisioning
```c
// BLE GATT characteristics for configuration
#define UUID_WIFI_SSID      0x0001
#define UUID_WIFI_PASSWORD  0x0002
#define UUID_NETWORK_KEY    0x0003
#define UUID_LOCATION       0x0004
#define UUID_SCHEDULE       0x0005
#define UUID_STATUS         0x0006

void ble_config_init(void);
void ble_config_start_advertising(void);
```

**Key Features:**
- BLE GATT server for smartphone config
- Encrypted pairing
- No app required (Web Bluetooth compatible)
- QR code for easy onboarding

#### NVS Storage
```c
void config_save(const char *key, void *data, size_t len);
void config_load(const char *key, void *data, size_t len);
void config_factory_reset(void);
```

## Communication Protocols

### Mesh Message Format
```
┌──────────┬──────────┬──────────┬─────────┬──────────┬──────────┐
│  Type    │  Source  │   Dest   │   TTL   │  Length  │ Payload  │
│  1 byte  │  2 bytes │  2 bytes │ 1 byte  │  1 byte  │ Variable │
└──────────┴──────────┴──────────┴─────────┴──────────┴──────────┘
```

### Heartbeat Message (every 30s)
```json
{
  "type": "heartbeat",
  "node_id": 1234,
  "uptime": 86400,
  "battery_soc": 75,
  "solar_power": 3.2,
  "light_state": true,
  "brightness": 128,
  "neighbors": [1235, 1236, 1237]
}
```

### Energy Status Message
```json
{
  "type": "energy",
  "node_id": 1234,
  "battery": {
    "voltage": 3.25,
    "current": 0.5,
    "soc": 75,
    "temperature": 28
  },
  "solar": {
    "voltage": 5.2,
    "current": 0.8,
    "power": 4.16
  },
  "needs_help": false,
  "can_help": true
}
```

## OTA Update System

### Update Flow
1. New firmware announced via mesh broadcast
2. Leader validates signature and distributes to all nodes
3. Each node downloads via mesh (chunked transfer)
4. Verification via SHA256 hash
5. Staged rollout (10% -> 50% -> 100%)
6. Automatic rollback on boot failure

### Security
- Ed25519 signature verification
- Encrypted transfer
- Version checking (no downgrade)
- Rollback partition for recovery

## Testing Strategy

### Unit Tests
- MPPT algorithm simulation
- Battery state machine
- Message serialization
- Consensus protocol

### Integration Tests
- Two-node mesh formation
- Multi-hop routing
- Energy sharing scenario
- OTA update process

### Hardware-in-Loop Tests
- Solar simulator input
- LED brightness verification
- Range testing
- Environmental stress

## Build and Flash

### Prerequisites
```bash
# Install ESP-IDF v5.1+
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh && . ./export.sh
```

### Build
```bash
cd firefly-firmware
idf.py set-target esp32c6
idf.py build
```

### Flash
```bash
idf.py -p /dev/ttyUSB0 flash monitor
```

### Configuration
```bash
idf.py menuconfig
# Navigate to: Firefly Configuration
#   - Set Thread network key
#   - Set channel (11-26)
#   - Set LED parameters
```

## Performance Targets

| Metric | Target | Measured |
|--------|--------|----------|
| Boot time | <3s | TBD |
| Mesh join time | <10s | TBD |
| Message latency (1 hop) | <50ms | TBD |
| Message latency (3 hop) | <200ms | TBD |
| Memory usage | <200KB | TBD |
| Flash usage | <1MB | TBD |
| CPU idle (light sleep) | >90% | TBD |

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | Jan 2026 | Initial architecture |

## Contributors

- Add your name here when you contribute!
