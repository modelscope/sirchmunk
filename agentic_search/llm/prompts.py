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
### Role: Search Optimization Expert
### Task:
Analyze the provided user query to extract core keywords. Output the result as a comma-separated list of optimized keywords.
Output **fine-grained** and **coarse-grained** keywords.
ONLY from the user query, do NOT add any external information.

### User Query:
{user_input}

### Optimized Keywords:
"""


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
