/**
 * Tests for Physical Barrier Detection Module
 */

import { detectPhysicalBarrier, PHYSICAL_KEYWORDS } from './physical-barrier';

// Test cases for detectPhysicalBarrier
const testCases = [
  {
    description: 'Task with "order" keyword',
    input: 'Order the PCB components from digikey',
    shouldDetect: true,
    expectedType: 'order_parts',
  },
  {
    description: 'Task with "assemble" keyword',
    input: 'Assemble the circuit board with all components',
    shouldDetect: true,
    expectedType: 'assemble',
  },
  {
    description: 'Task with "solder" keyword',
    input: 'Solder the headers to the main board',
    shouldDetect: true,
    expectedType: 'assemble',
  },
  {
    description: 'Task with "test with hardware" keyword',
    input: 'Test with hardware to verify the circuit works',
    shouldDetect: true,
    expectedType: 'test',
  },
  {
    description: 'Task with "measure" keyword',
    input: 'Measure the voltage output with a multimeter',
    shouldDetect: true,
    expectedType: 'measure',
  },
  {
    description: 'Task with "photograph" keyword',
    input: 'Photograph the final prototype setup',
    shouldDetect: true,
    expectedType: 'photograph',
  },
  {
    description: 'Task with "build prototype" keyword',
    input: 'Build prototype of the new hardware interface',
    shouldDetect: true,
    expectedType: 'order_parts',
  },
  {
    description: 'Task with "wire" keyword',
    input: 'Wire the battery to the main controller',
    shouldDetect: true,
    expectedType: 'assemble',
  },
  {
    description: 'Task with "install" keyword',
    input: 'Install the heat sink on the processor',
    shouldDetect: true,
    expectedType: 'assemble',
  },
  {
    description: 'Task with no physical keywords',
    input: 'Update the documentation for the API',
    shouldDetect: false,
    expectedType: undefined,
  },
  {
    description: 'Empty string',
    input: '',
    shouldDetect: false,
    expectedType: undefined,
  },
  {
    description: 'Task with "purchase" keyword',
    input: 'Purchase resistors and capacitors for the circuit',
    shouldDetect: true,
    expectedType: 'order_parts',
  },
  {
    description: 'Task with "calibrate" keyword',
    input: 'Calibrate the sensor with a known reference',
    shouldDetect: true,
    expectedType: 'test',
  },
  {
    description: 'Task with "inspect" keyword',
    input: 'Inspect the assembled board for any defects',
    shouldDetect: true,
    expectedType: 'measure',
  },
];

console.log('Running Physical Barrier Detection Tests...\n');

let passCount = 0;
let failCount = 0;

testCases.forEach((testCase) => {
  const result = detectPhysicalBarrier(testCase.input);

  const detected = result !== null;
  const typeMatch = detected ? result.type === testCase.expectedType : true;

  const passed = detected === testCase.shouldDetect && typeMatch;

  if (passed) {
    passCount++;
    console.log(`✓ PASS: ${testCase.description}`);
  } else {
    failCount++;
    console.log(`✗ FAIL: ${testCase.description}`);
    if (detected !== testCase.shouldDetect) {
      console.log(`  Expected detection: ${testCase.shouldDetect}, got: ${detected}`);
    }
    if (!typeMatch) {
      console.log(
        `  Expected type: ${testCase.expectedType}, got: ${result?.type}`
      );
    }
  }
});

console.log(`\n========================================`);
console.log(`Test Results: ${passCount} passed, ${failCount} failed`);
console.log(`========================================\n`);

// Display available keywords
console.log('Available Physical Keywords:');
console.log(PHYSICAL_KEYWORDS.join(', '));
console.log();
