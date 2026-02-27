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

### Output Format:
Output {num_levels} separate JSON-like dicts within their respective tags:

{output_format_example}

### User Query:
{{user_input}}

### {num_levels}-Level Keywords (Coarse to Fine):
"""


def generate_keyword_extraction_prompt(num_levels: int = 3) -> str:
    """
    Generate a dynamic keyword extraction prompt template based on the number of levels.
    
    The returned template still contains {{user_input}} placeholder that needs to be
    filled in by the caller.
    
    Args:
        num_levels: Number of granularity levels (default: 3)
    
    Returns:
        Prompt template string with {{user_input}} placeholder
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
    
    # Format the template with num_levels, descriptions, and examples
    # Note: {{user_input}} becomes {user_input} after this format call
    return QUERY_KEYWORDS_EXTRACTION.format(
        num_levels=num_levels,
        level_descriptions="\n\n".join(level_descriptions),
        output_format_example="\n\n".join(output_examples)
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
After generating the summary, evaluate whether this knowledge cluster is worth saving to the persistent cache based on:
1. Does the search result contain substantial, relevant information for the user input?
2. Is the content meaningful and not just error messages or "no information found"?
3. Are there sufficient evidences and context to answer the user's query?

If YES to all above, output "true"; otherwise output "false".

### Output Format
<SUMMARY>
[Generate the Markdown Briefing here]
</SUMMARY>
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
Return JSON:
- score (0-10):
  0-3: Completely irrelevant.
  4-7: Contains relevant keywords or context but no direct answer.
  8-10: Contains exact data, facts, or direct answer.
- reasoning: Short reasoning in the SAME language as the query.

JSON format only.
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


ROI_RESULT_SUMMARY = """
### Task
Analyze the provided {text_content} and generate a concise summary in the form of a Markdown Briefing.

### Constraints
1. **Language Continuity**: The output must be in the SAME language as the User Input.
2. **Format**: Use Markdown (headings, bullet points, and bold text) for high readability.
3. **Style**: Keep it professional, objective, and clear. Avoid fluff.

### Input Data
- **User Input**: {user_input}
- **Search Result Text**: {text_content}

### Output
[Generate the Markdown Briefing here]
"""
