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
            desc_text = "Multi-word phrases (2-3 words) that are likely to appear **verbatim** in the target document. Prioritize standard domain terminology (e.g. financial statement headings, technical section titles)"
            examples = '"capital expenditure", "net income", "accounts payable", "operating cash flow", "total revenue"'
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
5. **Verify before answering**: For numerical calculations, complete ALL computation steps in the SUMMARY section FIRST. Only write the PRECISE_ANSWER tag AFTER you have verified the final result. If you discover an error during computation, use the corrected value in PRECISE_ANSWER.
6. **Value precision**: Preserve the precision from the source data. If the source says "$8.70 billion" or "36.8%", report that exact value. Only round when converting between units (e.g., $381,603 thousands → "$382 million"). When the query specifies a rounding rule, follow it exactly.
7. **Best-effort answering**: Always attempt to answer based on available evidence. When the query requests a specific metric, ratio, or calculation, compute it from whatever relevant data is available — even if the data is partial. Do not refuse to calculate a metric solely because you believe it is unconventional or less applicable for a given entity type. Only mark SHOULD_ANSWER as "false" when the evidence is entirely unrelated to the query.

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
<SUMMARY>
[Generate the Markdown Briefing here with detailed analysis, supporting evidence, and full calculation steps. Complete all reasoning BEFORE the PRECISE_ANSWER tag.]
</SUMMARY>
<PRECISE_ANSWER>
[State ONLY the final verified answer. For yes/no questions, start with "Yes" or "No". For identification questions ("What is the largest…?"), state the name/label. For value questions, state the number with units (e.g. "0.83", "$1,832 million"). For calculations, this MUST reflect the result from your completed computation above. If the query is open-ended, write a one-sentence conclusion.]
</PRECISE_ANSWER>
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
5. **Verify before answering**: For numerical calculations, complete ALL computation steps in the SUMMARY section FIRST. Only write the PRECISE_ANSWER tag AFTER you have verified the final result. If you discover an error during computation, use the corrected value in PRECISE_ANSWER.
6. **Value precision**: Preserve the precision from the source data. If the source says "$8.70 billion" or "36.8%", report that exact value. Only round when converting between units (e.g., $381,603 thousands → "$382 million"). When the query specifies a rounding rule, follow it exactly.
7. **Best-effort answering**: Always attempt to answer based on available evidence. When the query requests a specific metric, ratio, or calculation, compute it from whatever relevant data is available — even if the data is partial. Do not refuse to calculate a metric solely because you believe it is unconventional or less applicable for a given entity type. Only mark SHOULD_ANSWER as "false" when the evidence is entirely unrelated to the query.

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
<SUMMARY>
[Generate the Markdown Briefing here with detailed analysis, supporting evidence, and full calculation steps. Complete all reasoning BEFORE the PRECISE_ANSWER tag.]
</SUMMARY>
<PRECISE_ANSWER>
[State ONLY the final verified answer. For yes/no questions, start with "Yes" or "No". For identification questions ("What is the largest…?"), state the name/label. For value questions, state the number with units (e.g. "0.83", "$1,832 million"). For calculations, this MUST reflect the result from your completed computation above. If the query is open-ended, write a one-sentence conclusion.]
</PRECISE_ANSWER>
<SHOULD_ANSWER>true/false</SHOULD_ANSWER>
<SHOULD_SAVE>true/false</SHOULD_SAVE>
"""


# ---------------------------------------------------------------------------
# Deep Structured Reasoning prompts
# ---------------------------------------------------------------------------

DEEP_SECTION_SELECT = """Given the user query and a document section map, select the sections most likely to contain the answer.

### User Query
{query}

### Document Section Map
{section_map}

### Instructions
1. Identify which sections contain data needed to answer the query.
2. For questions requiring computation (ratios, growth rates, comparisons), select ALL sections containing the required input data — even if you think some may be redundant.
3. Prefer sections containing structured data (tables, financial statements) over narrative sections.
4. For financial/annual report queries, ALWAYS include sections matching these types when available:
   - Income Statement / Consolidated Statements of Operations (revenue, expenses, net income)
   - Balance Sheet / Consolidated Balance Sheets (assets, liabilities, equity)
   - Cash Flow Statement / Consolidated Statements of Cash Flows (capex, operating cash flow)
   - Notes to Financial Statements (breakdowns, segment data, detailed schedules)
   - Management's Discussion and Analysis (context, trends, explanations)
5. Select 2-6 sections. When in doubt, select MORE rather than fewer — missing data causes answer failure.

### Output
Return ONLY a JSON array of section indices (0-based) from the map above:
[0, 3, 5]
"""


# ---------------------------------------------------------------------------
# DEEP mode query classification (Plan B)
# ---------------------------------------------------------------------------

DEEP_QUERY_CLASSIFY = """Classify this search query along two dimensions.

Query: {query}

1. **Complexity** — how many reasoning steps are needed:
   - "simple": Direct lookup of a single value (e.g. "What was revenue in FY2023?")
   - "moderate": Requires light computation from 1-2 data points (e.g. "What was the gross margin?")
   - "complex": Multi-step computation, multi-period comparison, or cross-entity analysis

2. **Intent** — what the user needs:
   - "lookup": Find and extract a specific stated value
   - "computation": Calculate a derived metric (ratio, growth rate, difference, average)
   - "comparison": Compare values across time periods, segments, or companies

Return ONLY valid JSON on a single line:
{{"complexity": "simple", "intent": "lookup"}}
"""

# ---------------------------------------------------------------------------
# Intent-specific synthesis prompts (Plan C)
# ---------------------------------------------------------------------------

ROI_LOOKUP_SYNTHESIS = """### Task
Extract the specific value requested from the evidence and present it clearly.

### Constraints
1. **Language Continuity**: The output must be in the SAME language as the User Input.
2. Find the value stated in the evidence. If the exact total is not stated but its components are clearly present, compute it by summing the components.
3. **Value precision**: Preserve the precision from the source data. If the source says "36.8%", report "36.8%", not "37%". If the source says "$8.70 billion", report "$8.70 billion", not "$9 billion". Only round when explicitly converting between units (e.g., $302,578 thousands → "$303 million"; $381,603 thousands → "$382 million"). When the query specifies a rounding rule (e.g., "round to one decimal place"), follow it exactly.
4. If multiple candidate values exist, select based on the closest match to the query's time period, entity, and metric.
5. Quote the source passage containing the value.
6. Only mark SHOULD_ANSWER as "false" when no relevant data exists in the evidence. Always prefer attempting an answer over refusing.
7. When the evidence contains relevant data but you feel uncertain, still attempt to answer.
8. **Answer format**:
   - For yes/no questions (e.g., "Has X increased?", "Did the company…?"), PRECISE_ANSWER must start with "Yes" or "No", followed by a brief qualifier if needed.
   - For identification questions (e.g., "What is the largest segment?", "Which company had the highest…?"), PRECISE_ANSWER should state the name/label, not the numeric value.
   - For value questions (e.g., "What was total revenue?"), PRECISE_ANSWER should state the numeric value with units.

### Input Data
- **User Input**: {user_input}
- **Evidence**: {text_content}

### Output Format
<SUMMARY>
**Source passage**: [Quote the exact text containing the answer]

**Extracted value**: [The specific value found]
</SUMMARY>
<PRECISE_ANSWER>[value only, e.g. "$1,832 million", "Yes, it increased by 5%", "Cloud Services segment"]</PRECISE_ANSWER>
<SHOULD_ANSWER>true/false</SHOULD_ANSWER>
<SHOULD_SAVE>true/false</SHOULD_SAVE>
"""

ROI_COMPUTATION_SYNTHESIS = """### Task
Answer the query by extracting data from the evidence and performing the required calculation.

### Constraints
1. **Language Continuity**: The output must be in the SAME language as the User Input.
2. Follow this STRICT sequence — do NOT skip any step:
   a) **DATA EXTRACTION**: List each required data point with its exact value and where you found it.
   b) **FORMULA**: State the formula needed (e.g. Gross Margin = (Revenue - COGS) / Revenue).
   c) **SUBSTITUTION**: Plug in the extracted values into the formula.
   d) **CALCULATION**: Show arithmetic step by step. For each step, write the operation and its result.
   e) **VERIFICATION**: Re-compute the final result independently to confirm.
3. **Value precision**: Preserve meaningful precision in computed results.
   - Dollar amounts: when converting units, round to the nearest whole number in the target unit. Example: $381,603 thousands → "$382 million". Otherwise preserve the precision of the input values.
   - Percentages: round to 1 decimal place by default. If the query says "round to one decimal place", follow exactly. If the query says "round to nearest whole number" or the context clearly calls for it, round to whole.
   - Ratios: round to 2 decimal places.
   - When the query specifies "round to X decimal places", follow that exactly.
4. **Units**: Convert all values to consistent units before computing.
5. If any required data point is missing, explicitly state what is missing and mark SHOULD_ANSWER as "false".
6. **Definition precision**: When computing financial ratios, use the broadest standard definition unless the query specifies otherwise. Quick ratio = (Current Assets - Inventories) / Current Liabilities. Asset turnover = Revenue / Average Total Assets.
7. **Answer format**: For yes/no questions, PRECISE_ANSWER must start with "Yes" or "No". For identification questions, state the name/label, not just the number.

### Input Data
- **User Input**: {user_input}
- **Evidence**: {text_content}

### Output Format
<SUMMARY>
## Data Extraction
| Data Point | Value | Source |
|---|---|---|
| [name] | [exact value] | [where found in evidence] |

## Calculation
**Formula**: [state formula]
**Step 1**: [operation] = [result]
**Step 2**: [operation] = [result]
**Verification**: [re-compute to confirm]
</SUMMARY>
<PRECISE_ANSWER>[final computed value only]</PRECISE_ANSWER>
<SHOULD_ANSWER>true/false</SHOULD_ANSWER>
<SHOULD_SAVE>true/false</SHOULD_SAVE>
"""

ROI_COMPARISON_SYNTHESIS = """### Task
Compare the requested values across the specified dimensions (time periods, entities, or segments).

### Constraints
1. **Language Continuity**: The output must be in the SAME language as the User Input.
2. Extract values for EACH comparison dimension from the evidence.
3. Present in a structured comparison table.
4. State the direction and magnitude of difference or change.
5. **Precision**: Use exact values from the evidence. When computing changes, show the arithmetic. Preserve the precision of source values. Only round when converting units (e.g., thousands → millions). Percentages: round to 1 decimal place by default.
6. **"Best performing"** means highest growth rate or change rate, not highest absolute value, unless the query explicitly says "largest" or "highest revenue".
7. If values for any comparison dimension are missing, state what is missing.

### Input Data
- **User Input**: {user_input}
- **Evidence**: {text_content}

### Output Format
<SUMMARY>
## Comparison
| Dimension | Value | Source |
|---|---|---|
| [period/entity] | [value] | [where found] |

## Analysis
**Direction**: [increased/decreased/stable]
**Magnitude**: [absolute and/or percentage change, with arithmetic shown]
</SUMMARY>
<PRECISE_ANSWER>[concise comparison result, e.g. "Increased from $1.2B to $1.5B (25% growth)"]</PRECISE_ANSWER>
<SHOULD_ANSWER>true/false</SHOULD_ANSWER>
<SHOULD_SAVE>true/false</SHOULD_SAVE>
"""

# ---------------------------------------------------------------------------
# Evidence completeness check (Plan D)
# ---------------------------------------------------------------------------

EVIDENCE_COMPLETENESS_CHECK = """Given the query and available evidence, determine whether all data points needed to answer are present.

### Query
{query}

### Query Type
{intent}

### Evidence (excerpt)
{evidence_excerpt}

### Instructions
1. Identify the specific data points required to answer this query.
2. Check whether each data point's actual value appears in the evidence.
3. A data point is FOUND only if its numeric/factual value is explicitly stated.

Return ONLY valid JSON on a single line:
{{"complete": true, "missing": []}}
or
{{"complete": false, "missing": ["short description of what is missing"]}}
"""

# ---------------------------------------------------------------------------
# Computation correction (Plan E)
# ---------------------------------------------------------------------------

COMPUTATION_CORRECTION = """Your previous calculation contained an arithmetic error. Please revise.

### Query
{query}

### Your Previous Answer
{original_answer}

### Detected Error
- Expression: {expression}
- Your result: {llm_result}
- Correct result: {correct_result}

Revise your answer using the correct arithmetic. Keep the same analysis structure.

<SUMMARY>
[Corrected analysis with fixed calculation]
</SUMMARY>
<PRECISE_ANSWER>[Corrected final value]</PRECISE_ANSWER>
<SHOULD_ANSWER>true</SHOULD_ANSWER>
<SHOULD_SAVE>true</SHOULD_SAVE>
"""

# ---------------------------------------------------------------------------
# Agentic retrieval prompts (DEEP mode)
# ---------------------------------------------------------------------------

DEEP_DATA_REQUIREMENTS = """Given the user's question, identify the specific data points needed to answer it.

### Question
{query}

### Question Type
{intent}

### Instructions
1. List each specific data point needed to answer this question (e.g., "Total Revenue for FY2022", "Accounts Payable as of fiscal year end 2019").
2. For each data point, identify the likely document section type where it would appear (e.g., "Income Statement", "Balance Sheet", "Cash Flow Statement", "Notes to Financial Statements", "Management Discussion and Analysis", "Segment Information").
3. If a calculation is required, state the exact formula with explicit variable names matching how they typically appear in financial statements. Example: "Quick Ratio = (Total Current Assets - Total Inventories) / Total Current Liabilities".
4. Identify the time period(s) required.
5. For comparison or identification questions (e.g., "What is the largest segment?", "Which year had the highest growth?"), note what dimensions need comparison.

Return ONLY valid JSON on a single line:
{{"data_points": ["data point 1", "data point 2"], "likely_sources": ["section type 1", "section type 2"], "formula": "explicit formula with variable names, or null", "time_period": "period or null"}}
"""

DEEP_PAGE_SELECT = """You are locating specific data in a document. Select pages to fetch.

### Question
{query}

### Data Still Needed
{data_requirements}

### Document Outline (with page ranges)
{section_map}

### Pages Already Fetched
{fetched_pages}

### Evidence Already Gathered
{evidence_summary}

### Instructions
- Reason about which sections contain the needed data based on section titles, summaries, and page ranges.
- Consider what data has already been gathered to avoid fetching redundant content.
- Financial statements (Income Statement, Balance Sheet, Cash Flow Statement) typically contain quantitative data needed for calculations.
- Sections with tables are often high-value for data extraction.
- Do NOT re-select pages listed in "Pages Already Fetched".
- Select 3-8 pages that are most likely to contain the missing data.
- When uncertain, prefer sections deeper in the document (financial statements are usually after narrative sections).

Return ONLY a JSON array of page numbers to fetch: [45, 46, 52, 53]
"""

DEEP_CHECK_REQUIREMENTS = """Check whether the evidence contains all required data points.

### Question
{query}

### Required Data Points
{data_points}

### Formula (if applicable)
{formula}

### Evidence
{evidence}

### Instructions
For each required data point, check if its actual numeric or factual value appears in the evidence. A data point is FOUND only if you can identify its specific value in the text.

Return ONLY valid JSON:
{{"complete": true, "found": [{{"point": "description", "value": "extracted value"}}], "missing": []}}
or
{{"complete": false, "found": [{{"point": "description", "value": "extracted value"}}], "missing": ["description of missing data point"]}}
"""

DEEP_TOC_ANALYSIS = """Analyze the following pages from the beginning of a document and extract its structural outline.

### Document Pages
{toc_page_text}

### Total Document Pages
{total_pages}

### Instructions
1. Look for a table of contents, section listing, or structural overview.
2. Extract every section entry with its title, starting page number, and hierarchy level.
3. Infer page_end from the start of the next section (use {total_pages} for the last section).
4. If page numbers appear as dot leaders (e.g. "Item 7. MD&A ........ 45"), extract the page number.
5. If no structural information can be extracted, return an empty array.

Return ONLY valid JSON — an array of section objects:
[{{"title": "Section Title", "page_start": 3, "page_end": 15, "level": 1}}, ...]

If no structure found, return: []
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


COMPILE_CLASSIFY_HEADINGS = """Classify each bold text line as either a **section heading** or **non-heading**.

A line is a *section heading* if it serves as the title of a major structural division of the document (chapter, section, subsection, exhibit, schedule, financial statement, note, etc.).
A line is *non-heading* if it is emphasis text, a label, a caption, a total/subtotal row, or any inline bold phrase that does not introduce a new document section.

For each heading, also assign a Markdown heading level (2–4):
- Level 2: top-level sections (e.g. financial statements, major chapters)
- Level 3: sub-sections (e.g. notes to financial statements, sub-chapters)
- Level 4: sub-sub-sections

Return ONLY a JSON array of objects for the lines that ARE headings.
Each object: {{"idx": <0-based index>, "level": <2|3|4>}}
If none are headings, return an empty array: []

Bold lines:
{candidates}"""


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
