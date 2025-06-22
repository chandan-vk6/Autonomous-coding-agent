from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit
from groq import Groq
import os
import re
import logging
import json
from datetime import datetime
import time
import difflib
from typing import List, Dict, Optional, Tuple

# --- LangChain and Pydantic Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:5000"], ping_timeout=120, ping_interval=60)

# --- Groq API Configuration ---
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "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"
)
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY environment variable not set")

client = Groq(api_key=GROQ_API_KEY)
session_contexts = {}

# --- Pydantic Models for Structured Output ---
class Issue(BaseModel):
    start_line: int = Field(..., description="The starting line number of the issue (1-indexed).")
    end_line: int = Field(..., description="The ending line number of the issue (1-indexed).")
    severity: str = Field(..., description="A string, either 'error', 'warning', or 'info'.")
    description: str = Field(..., description="A clear, user-friendly description of the problem.")

class Issues(BaseModel):
    issues: List[Issue] = Field(..., description="A list of issue objects.")


# --- Data Class for Interactive Bug Fixes ---
class BugFix:
    def __init__(self, start_line: int, end_line: int, original_code: str, 
                 fixed_code: str, bug_type: str, description: str):
        self.start_line = start_line
        self.end_line = end_line
        self.original_code = original_code.strip()
        self.fixed_code = fixed_code
        self.bug_type = bug_type
        self.description = description

# --- Utility Functions ---

def extract_json_from_response(response_text: str) -> Optional[Dict]:
    try:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match: return json.loads(json_match.group(1))
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if json_match: return json.loads(json_match.group(1))
        return json.loads(response_text)
    except (json.JSONDecodeError, AttributeError): return None

def get_leading_whitespace(line: str) -> str:
    match = re.match(r'^(\s*)', line)
    return match.group(1) if match else ''


@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')
# --- Socket.IO Handlers ---

@socketio.on('connect')
def on_connect():
    session_id = request.sid
    session_contexts[session_id] = {'history': [], 'current_code': '', 'last_interaction': datetime.now()}
    print(f'✅ Client connected: {session_id}')

@socketio.on('disconnect')
def on_disconnect():
    session_id = request.sid
    if session_id in session_contexts: del session_contexts[session_id]
    print(f'⚠ Client disconnected: {session_id}')

@socketio.on('prompt')
def on_prompt(data):
    try:
        session_id = request.sid
        prompt_text = data.get('content', '')
        mode = data.get('mode', 'generate')
        current_code = data.get('currentCode', '')
        bugfix_type = data.get('bugfixType', 'interactive')

        print(f"📥 {mode.upper()} ({bugfix_type if mode == 'bugfix' else 'N/A'}): {prompt_text[:100]}...")
        
        if session_id not in session_contexts: on_connect()
        
        context = session_contexts[session_id]
        context['current_code'] = current_code
        context['last_interaction'] = datetime.now()
        
        if mode == 'bugfix':
            if bugfix_type == 'auto':
                handle_auto_bugfix(session_id, current_code, prompt_text)
            else:
                handle_interactive_bugfix(session_id, current_code, prompt_text)
        elif mode == 'complete':
            handle_completion(session_id, prompt_text, current_code)
        else:
            handle_regular_generation(session_id, prompt_text, mode, current_code)
            
    except Exception as e:
        logging.error(f"❌ Unhandled Error in on_prompt: {e}", exc_info=True)
        emit('response', {'type': 'error', 'content': f"An unexpected server error occurred: {str(e)}"})

# --- Bugfix Mode 1: Auto-Fix (with Visuals) ---

def handle_auto_bugfix(session_id, current_code, user_context):
    try:
        if not current_code.strip():
            emit('response', {'type': 'error', 'content': "No code to analyze."})
            return

        emit('bugfix_status', {'stage': 'analyzing', 'message': 'Analyzing code for issues...'})
        issues_list = analyze_code_issues_with_langchain(current_code, user_context)
        
        if not issues_list:
            emit('bugfix_status', {'stage': 'complete', 'message': 'Analysis complete. No issues found.'})
            emit('response', {'type': 'no_fixes_found', 'content': 'Analysis complete. No issues found.'})
            return
        
        emit('bugfix_status', {'stage': 'fixing', 'message': f'Found {len(issues_list)} issues. Generating fix...'})
        fixed_code = generate_complete_fix(current_code, [issue.dict() for issue in issues_list], user_context)

        if not fixed_code:
            emit('response', {'type': 'error', 'content': "Failed to generate a valid fix."})
            emit('bugfix_status', {'stage': 'complete', 'message': 'Fix generation failed.'})
            return

        original_lines = current_code.splitlines()
        fixed_lines = fixed_code.splitlines()
        
        emit('bugfix_status', {'stage': 'fixing', 'message': 'Applying fixes with visual feedback...'})
        
        emit('start_visual_fix', {'total_lines': len(original_lines)})
        socketio.sleep(0.5)

        for i, line in enumerate(original_lines):
            line_num = i + 1
            emit('visual_fix_cursor', {'line': line_num})

            if i < len(fixed_lines) and line != fixed_lines[i]:
                emit('update_line', {'line_num': line_num, 'new_content': fixed_lines[i]})
            
            socketio.sleep(0.1)
            
        socketio.sleep(0.5)

        emit('final_code_sync', {'code': fixed_code})
        emit('bugfix_status', {'stage': 'complete', 'message': 'All issues fixed successfully!'})

    except Exception as e:
        logging.error(f"❌ Auto-Bugfix Error: {e}", exc_info=True)
        emit('response', {'type': 'error', 'content': f"Error during auto-bugfix: {str(e)}"})
        emit('bugfix_status', {'stage': 'complete', 'message': 'An error occurred.'})

def analyze_code_issues_with_langchain(code: str, user_context: str) -> List[Issue]:
    try:
        llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", groq_api_key=GROQ_API_KEY)
        structured_llm = llm.with_structured_output(Issues)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert Python code analyzer. Analyze the provided code for bugs, logical errors, or syntax issues. Format your response according to the provided schema."),
                ("human", "Analyze this code:\n\n```python\n{code}\n```\n\nUser Context: '{user_context}'")
            ]
        )
        chain = prompt | structured_llm
        response_model = chain.invoke({"code": code, "user_context": user_context or "General analysis."})
        return response_model.issues
    except Exception as e:
        logging.error(f"❌ LangChain Analysis Error: {e}", exc_info=True)
        return []

def generate_complete_fix(code, issues, user_context):
    try:
        issues_summary = "\n".join([f"- Line {issue.get('start_line', '?')}: {issue.get('description', 'N/A')}" for issue in issues])
        fix_prompt = f"""You are a surgical code correction expert. Fix the specific bugs in the Python code.
**Original Code:**
```python
{code}
```
**Identified Issues to Fix:**
{issues_summary}
**Instructions:** Correct ONLY the described bugs. Do not add features or refactor. Your output MUST be ONLY the complete, raw, corrected Python code."""
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0.0, max_tokens=4096
        )
        fixed_code = response.choices[0].message.content.strip()
        fixed_code = re.sub(r"^```(?:python)?\s*|\s*```$", "", fixed_code, flags=re.MULTILINE)
        return fixed_code.strip()
    except Exception as e:
        logging.error(f"❌ Fix Generation Error: {e}", exc_info=True)
        return None

# --- Bugfix Mode 2: Interactive Fix ---

def handle_interactive_bugfix(session_id, code, user_context):
    try:
        if not code.strip():
            emit('response', {'type': 'error', 'content': "No code provided for analysis."})
            return

        emit('bugfix_status', {'message': 'Starting comprehensive code analysis...'})
        response_data = analyze_code_with_direct_api(code, user_context)
        
        if not response_data:
            emit('response', {'type': 'error', 'content': "Failed to analyze code. Please try again."})
            return
        
        emit('bugfix_status', {'message': 'Processing analysis results...'})
        fixes = parse_bug_fixes(response_data, code.split('\n'))
        
        if not fixes:
            emit('response', {'type': 'no_fixes_found', 'content': 'Analysis complete. No bugs detected.'})
            return
        
        _, changes_metadata = apply_fixes_to_code(code, fixes)
        emit('visual_fix_diff', {'changes': changes_metadata})

    except Exception as e:
        logging.error(f"❌ Interactive Bugfix Error: {e}", exc_info=True)
        emit('response', {'type': 'error', 'content': f"Analysis error: {str(e)}"})

def analyze_code_with_direct_api(code: str, user_context: str) -> Optional[Dict]:
    numbered_code = '\n'.join([f"{i:2d}: {line}" for i, line in enumerate(code.split('\n'), 1)])
    system_prompt = """You are an expert Python code debugger. Analyze the code and identify ALL bugs.
CRITICAL INSTRUCTIONS:
1. Provide EXACT start and end line numbers for EACH bug.
2. `fixed_code` must be the complete corrected line(s) WITHOUT leading indentation.
3. Return response as VALID JSON only.
JSON FORMAT:
```json
{"analysis_summary": "Brief summary", "fixes": [{"start_line": num, "end_line": num, "bug_type": "type", "fixed_code": "code", "description": "desc"}]}
```"""
    user_prompt = f"Analyze this code:\n\n{numbered_code}\n\nUser context: {user_context}\n\nRespond ONLY with JSON."
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.1, max_tokens=4000
        )
        return extract_json_from_response(response.choices[0].message.content.strip())
    except Exception as e:
        logging.error(f"Error in direct API call: {e}")
        return None

def parse_bug_fixes(response_data: Dict, code_lines: List[str]) -> List[BugFix]:
    fixes = []
    if 'fixes' not in response_data: return fixes
    for fix_data in response_data.get('fixes', []):
        try:
            start_line, end_line = int(fix_data['start_line']), int(fix_data['end_line'])
            if not (1 <= start_line <= len(code_lines) and start_line <= end_line <= len(code_lines)): continue
            raw_fixed_code = fix_data.get('fixed_code', '')
            preserved_fixed_code = apply_indentation_preserving_fix(code_lines[start_line-1:end_line], raw_fixed_code)
            fix = BugFix(start_line, end_line, '\n'.join(code_lines[start_line-1:end_line]), preserved_fixed_code, fix_data.get('bug_type'), fix_data.get('description'))
            if fix.fixed_code.strip() != '\n'.join(code_lines[start_line-1:end_line]).strip(): fixes.append(fix)
        except (ValueError, KeyError): continue
    return remove_overlapping_fixes(sorted(fixes, key=lambda x: x.start_line))

def remove_overlapping_fixes(fixes: List[BugFix]) -> List[BugFix]:
    if not fixes: return []
    non_overlapping = [fixes[0]]
    for fix in fixes[1:]:
        if fix.start_line > non_overlapping[-1].end_line: non_overlapping.append(fix)
    return non_overlapping

def apply_indentation_preserving_fix(original_lines: List[str], fixed_code: str) -> str:
    if not original_lines or not fixed_code.strip(): return fixed_code
    base_indent = get_leading_whitespace(original_lines[0])
    return '\n'.join([base_indent + line for line in fixed_code.strip().split('\n')])

def apply_fixes_to_code(code: str, fixes: List[BugFix]) -> Tuple[str, List[Dict]]:
    lines = code.split('\n')
    changes_metadata = []
    for fix in reversed(fixes):
        changes_metadata.insert(0, {"start_line": fix.start_line, "end_line": fix.end_line, "original_lines": lines[fix.start_line-1:fix.end_line], "fixed_lines": fix.fixed_code.split('\n'), "description": fix.description, "bug_type": fix.bug_type})
    return "", changes_metadata

# --- New Completion Feature ---
def handle_completion(session_id, prompt_text, current_code):
    try:
        if not current_code.strip():
            emit('response', {'type': 'error', 'content': "There is no code to complete."})
            return

        system_prompt = """You are an expert code completion assistant.
The user will provide incomplete Python code, often with `pass` statements or missing logic.
Your task is to logically fill in the missing parts to create a complete, runnable script.
- Complete function bodies.
- Fill in missing arguments or variables.
- Correct minor syntax errors like a user's `**name**` to `__name__`.
- Ensure the code is syntactically correct and logically coherent.
IMPORTANT: Respond with ONLY the complete, final Python code. Do not add any explanations, comments, or ```python markers. Just the raw code."""

        user_prompt = f"Complete this code. Additional user requirements (if any): '{prompt_text}'\n\n```python\n{current_code}\n```"

        emit('bugfix_status', {'stage': 'analyzing', 'message': 'Analyzing for completion...'})

        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        completed_code = response.choices[0].message.content.strip()
        completed_code = re.sub(r"^```(?:python)?\s*|\s*```$", "", completed_code, flags=re.MULTILINE).strip()

        emit('bugfix_status', {'stage': 'fixing', 'message': 'Applying completions...'})
        
        original_lines = current_code.splitlines()
        completed_lines = completed_code.splitlines()
        matcher = difflib.SequenceMatcher(None, original_lines, completed_lines, autojunk=False)
        
        patches = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            patches.append({
                'start_line': i1 + 1,
                'end_line': i2,
                'new_text': '\n'.join(completed_lines[j1:j2])
            })
        
        # Emit patches in reverse order (bottom to top) to not mess up line numbers
        for patch in reversed(patches):
            emit('completion_patch', patch)
            socketio.sleep(0.1) 

        context = session_contexts[session_id]
        context['history'].append({'prompt': prompt_text, 'response': '[Code Completed]', 'mode': 'complete'})
        
        emit('response', {'type': 'end'})

    except Exception as e:
        logging.error(f"❌ Completion Error: {e}", exc_info=True)
        emit('response', {'type': 'error', 'content': f"An error occurred during code completion: {str(e)}"})
        emit('bugfix_status', {'stage': 'complete', 'message': 'An error occurred.'})

# --- Regular Generation and Context Management ---
def handle_regular_generation(session_id, prompt_text, mode, current_code):
    context = session_contexts[session_id]
    messages = build_context_messages(context, prompt_text, mode, current_code)
    try:
        stream = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages, temperature=0.2, max_tokens=2048, stream=True)
        inside_code_block = False
        accumulated_text = ""
        response_content = ""
        
        for chunk in stream:
            part = chunk.choices[0].delta.content or ""
            
            if not part:
                continue
                
            if "```" in part:
                if not inside_code_block:
                    inside_code_block = True
                    part = re.sub(r"```(?:python)?\s*", "", part)
                else:
                    inside_code_block = False
                    part = part.replace("```", "")
                    
            if inside_code_block or "```" not in accumulated_text:
                if part:
                    emit('response', {'type': 'stream', 'content': part})
                    response_content += part
                    socketio.sleep(0.02)
            
            accumulated_text += part # Clear buffer for next round

        # print(response_content)
        context['history'] = (context.get('history', []) + [{'prompt': prompt_text, 'response': response_content, 'mode': mode}])[-10:]
        emit('response', {'type': 'end'})
    except Exception as e:
        logging.error(f"❌ Generation Error: {e}")
        emit('response', {'type': 'error', 'content': f"API Error: {str(e)}"})

def build_context_messages(context, prompt_text, mode, current_code):
    system_prompt = ("You are an AI code generation assistant. Respond with ONLY raw Python code."
                     "Do not add any explanations, markdown, or triple backticks.")
    user_prompt = prompt_text
    messages = [{"role": "system", "content": system_prompt}]
    for interaction in context.get('history', [])[-3:]:
        messages.append({"role": "user", "content": interaction['prompt']})
        messages.append({"role": "assistant", "content": interaction['response']})
    messages.append({"role": "user", "content": user_prompt})
    return messages

if __name__ == '__main__':
    print("🚀 Starting Merged Code Analysis Server on http://localhost:5000")
    print("✨ Features: Interactive Diff, Auto-Fix Toggle, Conversation History, Code Completion")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
