# Inference and Opinion-Driven Dataset Generation

## Why This Matters
Dataset quality comes from **analytical depth**, not just information density.
A conversation exploring "Why does this matter?" or "When would this fail?" is far more valuable than one merely explaining "What is X?"

Inference-driven conversations:
- Build analytical thinking in the reader
- Explore complexity and nuance
- Develop judgment and critical evaluation
- Connect ideas across domains
- Reveal assumptions and hidden logic

## Core Inference Strategies

### 1. Pattern Recognition & Synthesis
Scan multiple pages for recurring themes, structures, and relationships:
- **Do certain concepts appear together?** Why? Is there a causal link?
- **Is there a progression from simple to complex?** Or from old to new?
- **Are there implicit hierarchies or dependencies?** What comes first? What builds on what?
- **What is repeated or emphasized?** Why might this be important?

**Conversation example**: Document discusses "climate change" on pages 3, 7, 15, 22
→ Instead of Q&A about each mention, create: "What is the relationship between the three mechanisms the document describes? How do they reinforce each other?"

### 2. Counterfactual Reasoning
Explore what would change under different conditions:
- "What if the opposite were true?" 
- "What would happen if [key assumption] didn't hold?"
- "How would the outcome differ if [factor] changed?"

**Conversation example**: Document describes why Company X succeeded
→ Create: "What if the market conditions were different? Would the same strategy have worked in a slower-growth period?"

### 3. Domain Bridging & Analogy
Connect the document's domain to adjacent or surprising fields:
- If it's about biology → How does it relate to chemistry, medicine, environmental science, philosophy?
- If it's about economics → How does it relate to psychology, history, politics?
- **Find surprising connections** that deepen understanding

**Conversation example**: Document about mechanical engineering
→ Create: "In what ways is designing a machine similar to designing a software system? What principles transfer?"

### 4. Stakeholder & Perspective Analysis
For events, processes, disagreements, cases:
- **Different perspectives**: How do affected groups view this differently?
- **Conflict of interest**: Who benefits? Who is harmed? What are competing priorities?
- **Power dynamics**: Why could one group enforce their preference over others?
- **Unintended consequences**: What second-order effects emerged?

**Conversation example**: Document describes a policy change
→ Create: "How might workers, employers, and government view this differently? What would each group prioritize?"

### 5. Temporal & Causal Analysis
Examine what changed and why:
- **Before/during/after**: What was the state before? What changed? What resulted?
- **Triggers vs. root causes**: Was this event triggered by a specific incident, or was it inevitable given underlying conditions?
- **Feedback loops**: Do outcomes reinforce their initial causes?
- **Path dependence**: Would different early decisions have led to different outcomes?

**Conversation example**: Document describes the rise of an industry
→ Create: "What conditions had to exist for this industry to emerge when it did? Could it have happened 50 years earlier?"

### 6. Logical Evaluation & Critique
Test the strength of arguments and claims:
- **Evidence quality**: What evidence supports this? How strong is it?
- **Alternative explanations**: Could different causes explain the same effect?
- **Scope**: Does this claim hold in all cases, or only some? What are the boundaries?
- **Coherence**: Is this consistent with other claims in the document? Where might tensions exist?

**Conversation example**: Document makes a causal claim
→ Create: "What alternative factors could explain the observed pattern? How would we distinguish between them?"

### 7. Assumption Surfacing & Testing
Make implicit premises explicit:
- "What assumptions must be true for this to work?"
- "If [assumption] is wrong, what changes?"
- "How confident should we be in [assumption]? What evidence exists?"

**Conversation example**: Document recommends a strategy
→ Create: "What must be true about human behavior for this strategy to succeed? Is that assumption reasonable?"

### 8. Prediction & Extrapolation
Think forward from current conditions:
- "If these trends continue, what happens in 5/10/50 years?"
- "What early warning signs would indicate a shift?"
- "What could disrupt this trajectory?"

**Conversation example**: Document analyzes current market trends
→ Create: "Based on these patterns, what should we expect in the next 5 years? What could surprise us?"

### 9. Trade-off & Value Analysis
Make explicit what is gained and lost:
- "Choosing X over Y means gaining __ but losing __"
- "In what contexts is X better? When is Y preferable?"
- "How do values or priorities influence which choice is 'best'?"

**Conversation example**: Document compares two approaches
→ Create: "If you had to choose between approach A (faster but riskier) and B (slower but safer), what factors would determine your choice?"

## Question Framing Guide

### Questions That Invite Inference (✅ GOOD)
- "Why is X important in the context of Y?"
- "How does concept A relate to concept B?" (open-ended, requires synthesis)
- "What would happen if [condition] changed?"
- "One could argue that X has major implications for... what are they?"
- "Is this claim necessarily true, or only sometimes? When would it fail?"
- "What assumptions must hold for this strategy to work?"
- "What would someone who disagrees with this argument say?"

### Questions That Are Surface-Level (❌ AVOID)
- "What does the text say about X?" (no inference needed)
- "What is X?" (simple definition lookup)
- "Is Y mentioned in the document?" (binary fact check)
- "Who did Z?" (simple information retrieval)
- "What are the benefits of X?" (direct quote from text)

## Opinion Generation Guide

### How to Frame Opinions
Opinions should be grounded in the document but go beyond it:

✅ **Grounded & analytical**: "Based on the mechanisms described, one could argue that X is more fundamental than Y because..."
✅ **Evaluative**: "While the document presents this as positive, a critical perspective might question..."
✅ **Predictive**: "The evidence suggests that within 5 years, this field will..."
✅ **Comparative**: "This approach is stronger than alternatives across most criteria, but weaker in [domain]..."

❌ **Baseless**: An opinion that contradicts the document without reason
❌ **Off-topic**: An opinion unrelated to the document's domain
❌ **Mere assertion**: "I think X is better" without supporting reasoning

## Assessment Rubric for Inference Quality

Rate each conversation:

| Aspect | Strong (✅) | Weak (❌) |
|--------|-----------|---------|
| **Requires reasoning** | Q&A demands synthesis across pages or logical inference | Q&A is answerable by reading single page literally |
| **Tests understanding** | Shows grasp of mechanisms, not just surface facts | Treats content as list of facts to recall |
| **Explores nuance** | Acknowledges conditions, edge cases, trade-offs | Presents as absolute or simple |
| **Original insight** | Reveals new perspective or connection | Paraphrases existing text |
| **Practical application** | Shows how ideas apply or fail in varied contexts | Abstract without grounding |
| **Critical thinking** | Questions assumptions, weighs evidence | Accepts claims at face value |

## Common Pitfalls to Avoid

1. **Over-relying on document surface**: Don't just ask what a sentence literally says; ask what it implies
2. **Ignoring contradictions**: If two claims seem to conflict, make that a conversation topic
3. **Missing context**: Always read 2-3 consecutive pages before writing a conversation to ensure you understand the broader context
4. **Vague inference**: "What can we learn from this?" is too abstract; be specific about what inference you're testing
5. **False balance**: Don't present unfounded alternatives as equally valid; ground all alternatives in document content
6. **Scope creep**: Keep conversations grounded in the document, even when making broader inferences
7. **Passive phrasing**: Use active, engaged language ("Why does this matter?" not "Is this relevant?")

## Integration with Web Search

When enabled, use web search to enhance inference:
- **Test predictions**: Search for real-world outcomes post-document publication
- **Find applications**: Search for how the principle has been applied in practice
- **Discover contradictions**: Search for cases that challenge the document's claims
- **Add contemporary data**: Supplement historical analysis with current examples

Example: Document (published 2015) predicts market trend
→ Use `minimax_web_search` to find: "Has this prediction materialized? What evidence from 2015-2026?"
