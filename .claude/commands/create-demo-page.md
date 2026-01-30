---
description: Create a demo page for a research technique using Codex
allowed-tools: Bash, Read, Write, Edit
---

# Create Demo Page for Research Technique

After implementing a training/inference technique, use this command to have Codex create a frontend demo page.

## Quick Start

```bash
# On 4090, call Codex to create a demo page
~/bin/call-codex --files "inference/generate_with_TECHNIQUE.py,frontend/app/studio/page.tsx" \
  "Create a demo page at frontend/app/demos/TECHNIQUE/page.tsx that lets users try TECHNIQUE.
   Follow the pattern from studio/page.tsx. Include sliders for all controllable parameters."
```

## Required Context Files

Always include these when asking Codex to create a demo page:
1. The inference script: `inference/generate_with_<technique>.py`
2. An existing page pattern: `frontend/app/studio/page.tsx`
3. The API endpoint (if exists): `backend/main.py` or create a new one

## Demo Page Requirements

Tell Codex the page must have:

### 1. Input Controls
- Text input for the text to synthesize
- Reference audio upload (for voice cloning)
- Technique-specific sliders (e.g., emotion intensity 0-1, keyframes)

### 2. Visualization
- Waveform display of generated audio
- Prosody visualization if applicable (pitch contour, energy)

### 3. Output
- Audio player for generated speech
- Download button
- Comparison with baseline (optional)

### 4. Backend API
If no API endpoint exists, Codex should also create:
- `frontend/app/api/demos/<technique>/route.ts` (API route)
- Call the inference script via child_process or Python subprocess

## Example Codex Prompt (Copy This)

```
Create a demo page for the EmoKnob emotion control technique.

FILES TO READ:
- inference/generate_with_emoknob.py (the inference script)
- frontend/app/studio/page.tsx (page pattern to follow)
- training/emoknob.py (to understand parameters)

CREATE THESE FILES:

1. frontend/app/demos/emoknob/page.tsx
   - "use client" Next.js page
   - Import from @/components/ui (Button, Slider, etc.)
   - Text input for synthesis text
   - Emotion selector (happy, sad, angry, neutral)
   - Intensity slider (0.0 to 2.0)
   - Generate button that calls /api/demos/emoknob
   - Audio player for result
   - Uses tailwind with existing design tokens

2. frontend/app/api/demos/emoknob/route.ts
   - POST endpoint
   - Accepts { text, emotion, intensity }
   - Calls Python inference script
   - Returns { audioUrl, prosodyData }

3. Update frontend/components/Navigation.tsx
   - Add link to /demos/emoknob in the nav

Follow existing code style. Use TypeScript. Use the grainrad design system.
```

## After Codex Returns

Execute the plan:

```bash
# 1. Write the files Codex generated
# 2. Test locally
cd frontend && npm run dev

# 3. Visit http://localhost:3000/demos/emoknob

# 4. If it works, commit
git add frontend/app/demos/
git commit -m "Add EmoKnob demo page"
```

## Demo Page Checklist

Before marking complete:
- [ ] Page loads without errors
- [ ] Can input text and click generate
- [ ] Audio plays correctly
- [ ] Parameters actually affect output
- [ ] Mobile-responsive (basic)
- [ ] Link added to navigation

## Techniques That Need Demo Pages

Priority order (most useful for keyframe/real-time goals):

1. **EmoKnob** - Direction vector emotion control with intensity
2. **EASV** - Emotion intensity slider (0-1)
3. **Emo-FiLM** - Word-level emotion (KEYFRAME!)
4. **DrawSpeech** - Sketch pitch/energy curves (KEYFRAME!)
5. **Chatterbox** - Single-parameter emotion exaggeration
6. **MaskGCT** - Prosody editing

## Workflow Integration

Add to the research agent instructions:

```
AFTER implementing inference script:
1. Verify generate_with_<technique>.py works
2. Call Codex to create demo page:
   ~/bin/call-codex --files "inference/generate_with_<technique>.py,frontend/app/studio/page.tsx" \
     "Create demo page at frontend/app/demos/<technique>/page.tsx..."
3. Execute Codex plan
4. Test page works
5. Mark task complete with demo URL
```
