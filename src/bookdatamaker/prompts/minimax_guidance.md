# IMPORTANT: search_text vs minimax_web_search
- search_text: Searches ONLY within this document. Use it to find where a keyword appears in the document pages.
- minimax_web_search: Searches the REAL INTERNET. Use it to find external examples, latest news, real-world cases, industry data.
They are completely different tools. You SHALL NOT confuse them.

# Web Search Enhancement (MANDATORY BEFORE EACH SUBMISSION)
You have minimax_web_search for INTERNET search. **You MUST call it before each submit_dataset.**

## Pre-Submission Search Workflow
Before calling submit_dataset, you SHALL:
1. Identify the key claim, concept, or event in your planned conversation
2. Call minimax_web_search with a targeted query to find:
   - Real-world examples, case studies, or empirical evidence supporting or challenging the claim
   - Relevant statistics, data, or expert opinions
   - Recent news or developments related to the topic
   - Counterarguments, caveats, or limitations noted by practitioners
3. Integrate the search findings into your conversation BEFORE submitting:
   - Use search results to enrich answers with concrete evidence
   - Add a "however..." or "on the other hand..." angle if search finds contradictions
   - Ground abstract principles in real-world examples found via search
   - Cite specific evidence: e.g. "研究发现...", "据报道...", "实际案例表明..."

## Search Quality Rules
- ❌ Do NOT submit a conversation without first searching for supporting/challenging evidence
- ❌ Do NOT use generic queries like "what is X" — be specific: "X failure cases", "X real-world application 2024"
- ✅ Search for angles that strengthen your analytical depth: causes, effects, controversies, comparisons
- ✅ One well-targeted search per submission is sufficient; do not over-search
- ✅ If minimax_understand_image is available and the page has images, analyze them as part of evidence gathering
