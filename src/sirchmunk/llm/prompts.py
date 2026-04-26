# Copyright (c) ModelScope Contributors. All rights reserved.
# flake8: noqa
# yapf: disable


SNAPSHOT_KEYWORDS_EXTRACTION = """
Analyze the following document and extract the most relevant and representative key phrases.
Prioritize terms that capture the core topics, central concepts, important entities (e.g., people, organizations, locations), and domain-specific terminology.
Exclude generic words (e.g., "the", "and", "result", "study") unless they are part of a meaningful multi-word phrase.
Limit the output to {max_num} concise key phrases, ranked by significance.
You **MUST** output the key phrases as a comma-separated list without any additional explanation or formatting.
You **MUST** adjust the language of the key phrases to be consistent with the language of the input document.

**Intput Document**:
{document_content}
"""


SNAPSHOT_TOC_EXTRACTION = """
Generate a Table of Contents (ToC) from the given document, adapting its depth and content density to the document’s inherent complexity.

Requirements:

1. Adaptive Hierarchy Depth: Dynamically set the depth between 3 to 5 levels, based on the document’s structural and semantic complexity (e.g., 3 levels for simple notices, 5 for technical specs).
2. Summarized Entries: Each ToC item must concisely summarize the section’s core content (10–25 words), not just repeat headings. Capture purpose, key actions, or critical info.
3. Faithfulness: Do not invent sections. Infer headings only from logical paragraph groupings if explicit titles are absent.
4. Format: Use Markdown nested lists with 2-space indents per level (e.g., - → - → -). Output ToC only—no preamble or commentary.

**Input Document**:
{document_content}
"""


KEYWORD_QUERY_PLACEHOLDER = "__SIRCHMUNK_USER_QUERY__"


QUERY_KEYWORDS_EXTRACTION = """
### Role: Search Optimization Expert & Information Retrieval Specialist

### Task:
Extract **{num_levels} sets** of keywords from the user query with **different granularities** to maximize search hit rate.

### Multi-Level Keyword Granularity Strategy:

Extract {num_levels} levels of keywords with progressively finer granularity:

{level_descriptions}

### IDF Value Guidelines:
- Estimate the **IDF (Inverse Document Frequency)** for each keyword based on its rarity in general corpus
- IDF range: **[0-10]** where:
  - 0-3: Very common terms (e.g., "the", "is", "data")
  - 4-6: Moderately common terms (e.g., "algorithm", "network")
  - 7-9: Rare/specific terms (e.g., "backpropagation", "xgboost")
  - 10: Extremely rare/specialized terms
- IDF values are **independent** of keyword level - focus on term rarity, not granularity

### Requirements:
- Each level should have 3-5 keywords
- Keywords must become progressively **finer-grained** from Level 1 to Level {num_levels}
- **Level 1**: Coarse-grained phrases/multi-word expressions
- **Level {num_levels}**: Fine-grained single words or precise technical terms
- ONLY extract from the user query context; do NOT add external information
- Ensure keywords at different levels are complementary, not redundant
- **Cross-lingual**: After all keyword levels, provide a small set (2-4) of the most important keywords translated to the other language (Chinese↔English). Only translate the core domain terms — skip generic words.

### Output Format:
Output {num_levels} separate JSON-like dicts within their respective tags, followed by a cross-lingual block:

{output_format_example}

<KEYWORDS_ALT>
{{"translated_keyword1": idf_value, "translated_keyword2": idf_value}}
</KEYWORDS_ALT>

### User Query:
{query_placeholder}

### {num_levels}-Level Keywords (Coarse to Fine):
"""


def generate_keyword_extraction_prompt(num_levels: int = 3) -> str:
    """
    Generate a dynamic keyword extraction prompt template based on the number of levels.
    
    The returned template still contains a stable placeholder token that
    needs to be replaced by the caller.
    
    Args:
        num_levels: Number of granularity levels (default: 3)
    
    Returns:
        Prompt template string with a query placeholder token
    """
    # Generate level descriptions with granularity focus
    level_descriptions = []
    for i in range(1, num_levels + 1):
        # Define granularity characteristics
        if i == 1:
            granularity = "Coarse-grained"
            desc_text = "Multi-word phrases, compound expressions, broader concepts"
            examples = '"machine learning algorithms", "data processing pipeline", "neural network training"'
        elif i == num_levels:
            granularity = "Fine-grained"
            desc_text = "Single words, precise terms, atomic concepts"
            examples = '"optimization", "gradient", "tensor", "epoch"'
        else:
            granularity = f"Medium-grained (Level {i})"
            desc_text = "2-3 word phrases or compound terms transitioning to single words"
            examples = '"deep learning", "batch normalization", "learning rate"'
        
        level_descriptions.append(
            f"**Level {i}** ({granularity}):\n"
            f"   - Granularity: {desc_text}\n"
            f"   - Example keywords: {examples}\n"
            f"   - Note: IDF values should reflect term rarity, not granularity level"
        )
    
    # Generate output format examples (avoiding f-string interpolation issues)
    output_examples = []
    for i in range(1, num_levels + 1):
        # Use double braces to escape them in the format string
        example_dict = '{{"keyword1": idf_value, "keyword2": idf_value, ...}}'
        output_examples.append(
            f"<KEYWORDS_LEVEL_{i}>\n{example_dict}\n</KEYWORDS_LEVEL_{i}>"
        )
    
    # Format the template with num_levels, descriptions, and examples.
    # The user query placeholder remains untouched and is replaced later
    # with a simple string replace to avoid a fragile second `.format()`.
    return QUERY_KEYWORDS_EXTRACTION.format(
        num_levels=num_levels,
        level_descriptions="\n\n".join(level_descriptions),
        output_format_example="\n\n".join(output_examples),
        query_placeholder=KEYWORD_QUERY_PLACEHOLDER,
    )


EVIDENCE_SUMMARY = """
## Role: High-Precision Information Synthesis Expert

## Task:
Synthesize a structured response based on the User Input and the provided Evidences.

### Critical Constraint:
1. **Language Consistency:** All output fields (<DESCRIPTION>, <NAME>, and <CONTENT>) MUST be written in the **same language** as the User Input.
2. **Ignore irrelevant noise:** Focus exclusively on information that directly relates to the User Input. If evidences contain conflicting or redundant data, prioritize accuracy and relevance.

### Input Data:
- **User Input:** {user_input}
- **Retrieved Evidences:** {evidences}

### Output Instructions:
1. **<DESCRIPTION>**: A high-level, concise synthesis of how the evidences address the user input.
   - *Constraint:* Maximum 3 sentences. Written in the language of {user_input}.
2. **<NAME>**: A ultra-short, catchy title or identifier for the description.
   - *Constraint:* Exactly 1 sentence, maximum 30 characters. Written in the language of {user_input}.
3. **<CONTENT>**: A detailed and comprehensive summary of all relevant key points extracted from the evidences.
   - *Constraint:* Written in the language of {user_input}.

### Output Format:
<DESCRIPTION>[Concise synthesis]</DESCRIPTION>
<NAME>[Short title]</NAME>
<CONTENT>[Detailed summary]</CONTENT>
"""


SEARCH_RESULT_SUMMARY = """
### Task
Analyze the provided {text_content} and generate a concise summary in the form of a Markdown Briefing.

### Constraints
1. **Language Continuity**: The output must be in the SAME language as the User Input.
2. **Format**: Use Markdown (headings, bullet points, and bold text) for high readability.
3. **Style**: Keep it professional, objective, and clear. Avoid fluff.

### Input Data
- **User Input**: {user_input}
- **Search Result Text**: {text_content}

### Quality Evaluation
After generating the summary, make TWO decisions:
1) whether the query can be answered from the provided evidence;
2) whether this knowledge cluster is worth saving to persistent cache.

Evaluate based on:
1. Does the search result contain substantial, relevant information for the user input?
2. Is the content meaningful and not just error messages or "no information found"?
3. Are there sufficient evidences and context to answer the user's query?

- <SHOULD_ANSWER>: output "true" if the evidence contains relevant information that can help answer the query, even if it requires reasoning, computation, or interpretation. Only output "false" if the evidence is clearly irrelevant or contains no useful information for the query.
- <SHOULD_SAVE>: output "true" only if the evidence is sufficient AND the result is worth caching.
- If evidence is insufficient or irrelevant, both SHOULD_ANSWER and SHOULD_SAVE MUST be "false".

### Output Format
<SUMMARY>
[Generate the Markdown Briefing here]
</SUMMARY>
<SHOULD_ANSWER>true/false</SHOULD_ANSWER>
<SHOULD_SAVE>true/false</SHOULD_SAVE>
"""


EVALUATE_EVIDENCE_SAMPLE = """
You are a document retrieval assistant. Please evaluate if the text snippet contains clues to answer the user's question.

### Language Constraint:
Detect the language of the "Query" and provide the "reasoning" and "output" in the same language (e.g., if the query is in Chinese, the reasoning must be in Chinese).

### Inputs:
Query: "{query}"

Text Snippet (Source: {sample_source}):
"...{sample_content}..."

### Output Requirement:
Return a single JSON object (no extra text):
- score (0-10):
  0-3: Completely irrelevant.
  4-7: Contains relevant keywords or context but no direct answer.
  8-10: Contains exact data, facts, or direct answer.
- reasoning: Short reasoning in the SAME language as the query.

Example: {{"score": 7, "reasoning": "Contains relevant context about the topic."}}
"""


DETECT_DOC_INTENT = """Classify the user query below.

Determine whether the user wants to perform a **whole-document operation** on
file(s) they have provided — for example: summarize, analyze, translate, explain,
review, extract key points, rewrite, or any other operation that requires reading
the entire document rather than searching for a specific piece of information.

### User Query
{user_input}

### Output
Return a single JSON object, no extra text:
- If this is a whole-document operation:  {{"doc_level": true, "op": "<operation>"}}
  where <operation> is one of: summarize, analyze, translate, explain, extract, review, or a short free-form verb.
- If this is a specific search / retrieval query: {{"doc_level": false}}
"""


DIRECT_DOC_ANALYSIS = """
### Role: Document Analysis Expert

### Task
Analyze the provided document(s) and respond to the user's question or instruction
based strictly on the document content.

### Constraints
1. **Language Continuity**: The output MUST be in the **same language** as the User Input.
2. **Format**: Use Markdown (headings, bullet points, bold text) for readability.
3. **Faithfulness**: Base your response strictly on the provided content. Do not fabricate information.
4. If the content has been sampled (indicated by `[...content sampled...]` markers),
   acknowledge that your analysis is based on excerpts and may miss details.

### Document Content
{documents}

### User Input
{user_input}
"""


DOC_SUMMARY = """
### Role: Document Summarization Expert

### Task
Generate a comprehensive summary of the provided document(s) in response to the user's request.

### Constraints
1. **Language Continuity**: Output MUST be in the **same language** as the User Input.
2. **Format**: Use Markdown with clear headings, bullet points, and bold text for readability.
3. **Faithfulness**: Base your summary strictly on the provided content. Do not fabricate information.
4. **Conciseness**: Capture the key points, main arguments, conclusions, and important details.
5. If content has been sampled (indicated by `[...content sampled...]` markers),
   note that the summary is based on excerpts.

### Document Content
{documents}

### User Input
{user_input}

### Output
Provide a well-structured Markdown summary.
"""


DOC_CHUNK_SUMMARY = """
### Task
Summarize the following document chunk concisely, preserving key points, arguments, and important details.

### Constraints
1. **Language Continuity**: Output MUST be in the **same language** as the User Input.
2. **Conciseness**: Distill to the essential points only.

### Document Chunk
{chunk}

### User Input
{user_input}

### Output
Return a concise summary of this chunk.
"""


DOC_MERGE_SUMMARIES = """
### Task
Merge the following partial summaries into a single, coherent, comprehensive summary.

### Constraints
1. **Language Continuity**: Output MUST be in the **same language** as the User Input.
2. **Format**: Use Markdown with clear headings, bullet points, and bold text.
3. **Deduplication**: Remove redundant points across partial summaries.
4. **Coherence**: Produce a unified document, not a list of separate summaries.

### Partial Summaries
{summaries}

### User Input
{user_input}

### Output
Provide a well-structured Markdown summary.
"""


HISTORY_RELEVANCE_CHECK = """Determine whether the conversation history is topically relevant to the latest user message.

### Conversation History (last few turns)
{history}

### Latest User Message
{message}

### Rules
- Output JSON only: {{"relevant": true}} or {{"relevant": false}}
- "relevant" = true if the history and the latest message share the same topic, continue the same discussion, or the latest message references context from the history (e.g. pronouns, follow-up questions).
- "relevant" = false if the latest message introduces a completely new, unrelated topic with no dependency on prior context."""


QUERY_REWRITE = """Given the conversation history and the latest user message, rewrite the user message into a standalone search query that captures the full intent without relying on prior context. If the message is already self-contained, return it unchanged.

### Conversation History
{history}

### Latest User Message
{message}

### Output
Return ONLY the rewritten query, nothing else. Keep the same language as the user message."""


FAST_QUERY_ANALYSIS = """Classify the user query and, if it is a document/file search query, extract search terms at two granularity levels for a ripgrep file search.

### User Query
{user_input}

### Output
Return JSON only, no extra text:
{{"type": "search", "primary": ["compound phrase"], "fallback": ["term1", "term2"], "idf": {{"compound phrase": 8.0, "term1": 2.5, "term2": 6.0}}, "primary_alt": [], "fallback_alt": [], "file_hints": [], "intent": "..."}}

Rules:
- **type**: "search" if the query requires retrieving information from files or documents; "chat" if it is a greeting, small talk, identity question, or any other conversational message that does NOT need file retrieval. When type is "chat", set primary and fallback to empty arrays and put a brief natural reply (same language as the query) in "response". "summary" if the user wants to summarize, review, or analyze entire documents without searching for specific information — set primary/fallback to empty arrays.
- **primary**: 1 compound phrase (2-3 words) most likely to appear **verbatim** in the target document. Tried first.
- **fallback**: 1-3 single-word atomic terms from the primary phrase. Tried only if primary misses.
- **primary_alt / fallback_alt**: Cross-lingual equivalents of primary/fallback. If the query is in Chinese, provide English translations; if in English, provide Chinese translations. Only translate the most critical 1-2 terms. Empty arrays if no meaningful translation exists.
- **file_hints**: filename fragments or glob patterns ONLY if clearly implied; empty array otherwise.
- **intent**: one sentence describing the query intent.
- **idf**: Estimated Inverse Document Frequency weight (1.0-10.0 scale) for EVERY keyword in primary, fallback, primary_alt, and fallback_alt. Higher values (7-10) for rare/specific/domain terms; lower values (1-3) for common/generic words. Estimate based on general corpus frequency.

Example: query "How does transformer attention work?"
→ {{"type": "search", "primary": ["transformer attention"], "fallback": ["attention", "transformer"], "idf": {{"transformer attention": 8.5, "attention": 3.0, "transformer": 5.0, "注意力机制": 8.0, "注意力": 3.5, "变换器": 6.0}}, "primary_alt": ["注意力机制"], "fallback_alt": ["注意力", "变换器"], "file_hints": [], "intent": "understand transformer attention mechanism"}}

Example: query "认证机制怎么实现"
→ {{"type": "search", "primary": ["认证机制"], "fallback": ["认证", "鉴权"], "idf": {{"认证机制": 7.5, "认证": 3.0, "鉴权": 7.0, "authentication": 5.5, "auth": 3.0}}, "primary_alt": ["authentication"], "fallback_alt": ["auth"], "file_hints": [], "intent": "了解认证机制的实现方式"}}

Example: query "你好"
→ {{"type": "chat", "primary": [], "fallback": [], "idf": {{}}, "primary_alt": [], "fallback_alt": [], "file_hints": [], "intent": "greeting", "response": "你好！我是 Sirchmunk，一个智能文档搜索助手。有什么可以帮你的？"}}

Example: query "总结这几篇文档"
→ {{"type": "summary", "primary": [], "fallback": [], "idf": {{}}, "primary_alt": [], "fallback_alt": [], "file_hints": [], "intent": "summarize documents"}}
"""


FAST_QUERY_ANALYSIS_WITH_CATALOG = """Classify the user query, extract search terms, AND select the most relevant document(s) from the compiled index.

### User Query
{user_input}

### Compiled Document Index
{document_listing}

### Output
Return JSON only, no extra text:
{{"type": "search", "primary": ["compound phrase"], "fallback": ["term1", "term2"], "idf": {{"compound phrase": 8.0, "term1": 2.5}}, "primary_alt": [], "fallback_alt": [], "file_hints": [], "intent": "...", "selected_docs": [0, 2], "doc_confidence": "high"}}

Rules:
- **type**: "search" if the query requires retrieving information from files or documents; "chat" if it is a greeting, small talk, or conversational message — set primary/fallback to empty arrays, put a brief reply in "response". "summary" if the user wants to summarize entire documents.
- **primary**: 1 compound phrase (2-3 words) most likely to appear **verbatim** in the target document.
- **fallback**: 1-3 single-word atomic terms. Tried only if primary misses.
- **primary_alt / fallback_alt**: Cross-lingual equivalents (Chinese↔English). Only the most critical 1-2 terms.
- **file_hints**: filename fragments or glob patterns ONLY if clearly implied; empty array otherwise.
- **intent**: one sentence describing the query intent.
- **idf**: IDF weight (1.0-10.0) for EVERY keyword. Higher for rare terms.
- **selected_docs**: Index numbers (from the Compiled Document Index above) of the 1-3 most relevant documents for this query. Consider BOTH the filename and the summary. Choose documents whose content is most likely to answer the query.
- **doc_confidence**: "high" if you are very confident the selected documents contain the answer; "medium" if likely but uncertain; "low" if guessing.
"""


ROI_RESULT_SUMMARY = """
### Task
Analyze the provided {text_content} and generate a concise summary in the form of a Markdown Briefing.

### Constraints
1. **Language Continuity**: The output must be in the SAME language as the User Input.
2. **Format**: Use Markdown (headings, bullet points, and bold text) for high readability.
3. **Style**: Keep it professional, objective, and clear. Avoid fluff.
4. **Precision**: When the query asks for a specific value, ratio, number, percentage, or yes/no determination, you MUST compute it and state the precise result. Show key calculation steps when applicable.

### Input Data
- **User Input**: {user_input}
- **Search Result Text**: {text_content}

### Quality Evaluation
After generating the summary, make TWO decisions:
1) whether the query can be answered from the provided evidence;
2) whether this result is worth caching.

Evaluate based on:
1. Does the search result contain substantial, relevant information for the user input?
2. Is the content meaningful and not just error messages or "no information found"?
3. Are there sufficient evidences and context to answer the user's query?

- <SHOULD_ANSWER>: output "true" if the evidence contains relevant information that can help answer the query, even if it requires reasoning, computation, or interpretation. Only output "false" if the evidence is clearly irrelevant or contains no useful information for the query.
- <SHOULD_SAVE>: output "true" only if the evidence is sufficient AND the result is worth caching.
- If evidence is insufficient or irrelevant, both SHOULD_ANSWER and SHOULD_SAVE MUST be "false".

### Output Format
<PRECISE_ANSWER>
[If the query asks for a specific value, ratio, number, or factual answer, state ONLY the direct answer here (e.g. "0.83", "$1,832 million", "Yes", "Increased from 0.67 to 0.69"). If the query is open-ended, write a one-sentence conclusion.]
</PRECISE_ANSWER>
<SUMMARY>
[Generate the Markdown Briefing here with detailed analysis and supporting evidence]
</SUMMARY>
<SHOULD_ANSWER>true/false</SHOULD_ANSWER>
<SHOULD_SAVE>true/false</SHOULD_SAVE>
"""

ROI_RESULT_SUMMARY_WITH_CONTEXT = """
### Task
Analyze the provided evidence and generate a concise summary in the form of a Markdown Briefing.
Leverage the document context below for better understanding of the source material's structure and purpose.

### Constraints
1. **Language Continuity**: The output must be in the SAME language as the User Input.
2. **Format**: Use Markdown (headings, bullet points, and bold text) for high readability.
3. **Style**: Keep it professional, objective, and clear. Avoid fluff.
4. **Precision**: When the query asks for a specific value, ratio, number, percentage, or yes/no determination, you MUST compute it and state the precise result. Show key calculation steps when applicable.

### Document Context
{document_context}

### Input Data
- **User Input**: {user_input}
- **Search Result Text**: {text_content}

### Quality Evaluation
After generating the summary, make TWO decisions:
1) whether the query can be answered from the provided evidence;
2) whether this result is worth caching.

Evaluate based on:
1. Does the search result contain substantial, relevant information for the user input?
2. Is the content meaningful and not just error messages or "no information found"?
3. Are there sufficient evidences and context to answer the user's query?

- <SHOULD_ANSWER>: output "true" if the evidence contains relevant information that can help answer the query, even if it requires reasoning, computation, or interpretation. Only output "false" if the evidence is clearly irrelevant or contains no useful information for the query.
- <SHOULD_SAVE>: output "true" only if the evidence is sufficient AND the result is worth caching.
- If evidence is insufficient or irrelevant, both SHOULD_ANSWER and SHOULD_SAVE MUST be "false".

### Output Format
<PRECISE_ANSWER>
[If the query asks for a specific value, ratio, number, or factual answer, state ONLY the direct answer here (e.g. "0.83", "$1,832 million", "Yes", "Increased from 0.67 to 0.69"). If the query is open-ended, write a one-sentence conclusion.]
</PRECISE_ANSWER>
<SUMMARY>
[Generate the Markdown Briefing here with detailed analysis and supporting evidence]
</SUMMARY>
<SHOULD_ANSWER>true/false</SHOULD_ANSWER>
<SHOULD_SAVE>true/false</SHOULD_SAVE>
"""


# ---------------------------------------------------------------------------
# Knowledge Compile prompts
# ---------------------------------------------------------------------------

COMPILE_TREE_STRUCTURE = """Analyze the following document and identify its natural hierarchical structure (chapters, sections, subsections).

### Document Content (may be truncated)
{document_content}

### Output Requirements
Return a JSON array of top-level sections. Each section object must have:
- "title": Section heading or descriptive title
- "summary": 1-2 sentence summary of the section content
- "start_marker": A short text string (5-15 words) that appears verbatim at the start of this section in the document
- "end_marker": A short text string that appears at the start of the NEXT section (empty for the last section)

Maximum {max_sections} sections. Identify only the most significant structural boundaries.

### Output Format
Return ONLY a JSON array, no extra text:
[
  {{"title": "...", "summary": "...", "start_marker": "...", "end_marker": "..."}},
  ...
]
"""


COMPILE_SYNTHESIZE_SUMMARY = """Synthesize a comprehensive document summary from the following section summaries.

### Section Summaries
{sections}

### Output
Provide a unified, coherent summary in 3-8 sentences that captures the document's overall topic, key arguments, and conclusions. Do not simply list the sections — weave them into a natural narrative.
Write in the same language as the section summaries."""


COMPILE_DOC_SUMMARY = """Summarize the following document concisely, capturing the key topics, arguments, conclusions, and important details.

### File: {file_name}

### Document Content (may be truncated)
{document_content}

### Output
Provide a comprehensive summary in 3-8 sentences. Focus on:
1. What is this document about (main topic/purpose)
2. Key findings, arguments, or conclusions
3. Important details, data points, or methodologies

Write the summary in the same language as the document content."""


COMPILE_TOPIC_EXTRACTION = """Extract the 3-5 most important topics, concepts, or entities from the following summary.

### Summary
{summary}

### Output
Return ONLY a JSON array of topic strings, no extra text:
["topic1", "topic2", "topic3"]

Rules:
- Each topic should be 1-4 words
- Prefer specific, domain-relevant terms over generic ones
- Use the same language as the summary"""


COMPILE_MERGE_KNOWLEDGE = """You are merging new information into an existing knowledge cluster.

### Existing Knowledge
{existing_content}

### New Information
{new_summary}

### Task
Produce an updated, unified summary that:
1. Preserves all important information from the existing knowledge
2. Integrates the new information, avoiding redundancy
3. Highlights any contradictions or complementary perspectives
4. Maintains a coherent, well-structured narrative

### Output
Return ONLY the merged summary text (no extra tags or metadata). Keep the same language as the inputs."""
