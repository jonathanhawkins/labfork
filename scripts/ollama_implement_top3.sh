#!/bin/bash
# Implement top 3 approaches using FREE Ollama

echo "🦙 Implementing Top 3 Prosody Approaches with FREE AI"
echo "======================================================"
echo ""

APPROACHES=(
    "train_sparse_keyframe.py:Sparse keyframe prosody control"
    "train_ece_tts_easv.py:EASV intensity control"
    "train_draw_speech.py:User-drawn pitch curves"
)

for approach in "${APPROACHES[@]}"; do
    IFS=':' read -r script description <<< "$approach"
    
    echo "📝 Implementing: $script"
    echo "   Description: $description"
    echo ""
    
    # Step 1: Ollama reads existing implementation
    echo "   [1/4] Reading existing code with Ollama..."
    
    # Step 2: Ollama generates summary
    echo "   [2/4] Generating implementation plan..."
    ollama run qwen2.5-coder:14b "Read training/$script and create a 5-step implementation plan for integrating with V7 baseline" > /tmp/plan_$script.txt
    
    # Step 3: Show plan to user
    echo "   [3/4] Plan generated:"
    cat /tmp/plan_$script.txt | head -20
    
    # Step 4: User decides
    echo ""
    read -p "   [4/4] Proceed with implementation? (y/N) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   ✅ Will implement with Ollama (FREE)"
        echo "   💾 Plan saved to: evaluation/plans/$script.plan.txt"
        mkdir -p evaluation/plans
        mv /tmp/plan_$script.txt evaluation/plans/
    else
        echo "   ⏭️  Skipped"
    fi
    
    echo ""
done

echo "💰 Cost so far: $0 (all FREE with Ollama!)"
echo ""
echo "Next steps:"
echo "1. Review plans in evaluation/plans/"
echo "2. Let Ollama implement each approach"
echo "3. Use Codex ONLY for final review (~$10-20 total)"
