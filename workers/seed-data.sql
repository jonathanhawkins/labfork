-- Firefly Network Seed Data

-- Insert Project
INSERT INTO projects (id, name, slug, status, config)
VALUES (
  'firefly-network',
  'The Firefly Network',
  'firefly-network',
  'active',
  '{"branding":{"primaryColor":"#FFB84D","accentColor":"#1a1a2e"},"description":"Build solar-powered mesh lights that bring illumination, power, and connectivity to communities worldwide"}'
);

-- Insert Agents
INSERT INTO agent_state (agent_id, project_id, persona, memory, status, tokens_used)
VALUES
  ('spark', 'firefly-network', '{"name":"Spark","role":"Solar Energy Specialist","avatar":"⚡","color":"#F59E0B"}', '{}', 'idle', 0),
  ('mesh', 'firefly-network', '{"name":"Mesh","role":"Network Architect","avatar":"🕸️","color":"#8B5CF6"}', '{}', 'idle', 0),
  ('lumen', 'firefly-network', '{"name":"Lumen","role":"Light Engineer","avatar":"💡","color":"#10B981"}', '{}', 'idle', 0);

-- Insert Initial Tasks
INSERT INTO tasks (id, project_id, title, description, status, priority, requires_physical, progress)
VALUES
  ('task-mppt-research', 'firefly-network', 'Research MPPT algorithms for small solar panels', 'Research Maximum Power Point Tracking algorithms optimized for small-scale solar applications (<20W panels).', 'pending', 1, 0, 0),
  ('task-thread-mesh', 'firefly-network', 'Implement Thread mesh protocol on ESP32-C6', 'Implement OpenThread stack on ESP32-C6 for mesh networking with self-healing topology.', 'pending', 2, 0, 0),
  ('task-led-pwm', 'firefly-network', 'Design LED PWM driver with logarithmic dimming', 'Design PWM-based LED driver with human-perceived linear dimming using logarithmic curves.', 'pending', 3, 0, 0),
  ('task-battery-bms', 'firefly-network', 'Create battery management system for LiFePO4', 'Implement BMS with cell balancing, temperature monitoring, and protection circuits.', 'pending', 2, 0, 0),
  ('task-swarm-consensus', 'firefly-network', 'Implement swarm consensus algorithm', 'Design distributed consensus mechanism for mesh network decisions.', 'pending', 4, 0, 0),
  ('task-pcb-design', 'firefly-network', 'Design PCB schematic v1', 'Create initial PCB schematic integrating ESP32-C6, power management, and LED drivers.', 'pending', 5, 1, 0),
  ('task-order-parts', 'firefly-network', 'Order prototype components', 'Order components from BOM for initial prototype build.', 'blocked', 6, 1, 0),
  ('task-assemble-proto', 'firefly-network', 'Assemble first prototype', 'Hand-assemble first prototype unit for testing.', 'blocked', 7, 1, 0);
