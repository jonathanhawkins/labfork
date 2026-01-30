# Domain Plugin Template

This is a starter template for creating custom domain plugins.

## Quick Start

1. Copy this directory:
   ```bash
   cp -r .domains/_template .domains/your-domain-name
   ```

2. Edit `domain.yaml`:
   - Change `slug` to your domain name (lowercase, hyphens only)
   - Update `name` and `description`
   - Customize colors in `branding`
   - Add relevant `scene.props`
   - Configure `research` categories and keywords
   - Define `evaluation` metrics

3. Customize prompt templates in `prompts/`:
   - `research.md` - For web research agents
   - `implementation.md` - For coding agents
   - `evaluation.md` - For evaluation agents

4. Update activity definitions in `activities/`:
   - Modify existing activities or add new ones
   - Set appropriate icons and visualization settings

5. Test your domain:
   ```bash
   # Start the dev server
   npm run dev

   # Check the API
   curl http://localhost:3003/api/domain/your-domain-name

   # View in browser
   open http://localhost:3003/lab?domain=your-domain-name
   ```

## Files Overview

```
_template/
├── domain.yaml          # Main configuration
├── README.md            # This file
├── prompts/
│   ├── research.md      # Research agent prompt
│   ├── implementation.md # Implementation agent prompt
│   └── evaluation.md    # Evaluation agent prompt
└── activities/
    ├── research.yaml    # Research activity config
    ├── implementation.yaml # Implementation activity config
    ├── evaluation.yaml  # Evaluation activity config
    └── idle.yaml        # Idle state config
```

## Documentation

See the full guide: [Domain Plugin Development Guide](../../docs/DOMAIN_PLUGIN_GUIDE.md)
