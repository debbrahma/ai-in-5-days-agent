import sys
import os
import json
import re
import subprocess
import tempfile
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY environment variable is not set!")
    sys.exit(1)

# Initialize client using developer key and v1beta to access 3.x models
client = genai.Client(
    api_key=api_key, 
    vertexai=False,
    http_options=HttpOptions(api_version="v1beta")
)

SYSTEM_INSTRUCTION = """
You are "OptiMind", an elite Operations Research (OR) and Decision Intelligence Agent. Your job is to translate unstructured business problems into Python code using the Google OR-Tools library, solve them, and present the results in business-friendly terms.

### Operational Principles:
1. Mathematical Extraction: Extract Sets, Decision Variables, Constraints, and Objective function from the user's natural language description.
2. Dynamic Code Formulation: Write complete, runnable, and syntactically correct Python code using ortools.linear_solver.pywraplp or ortools.sat.python.cp_model.
3. Execution Safety: 
   - Never write code that accesses the internet, deletes files, or imports unauthorized modules.
   - Only import sys, json, and ortools.
   - All results MUST be printed to stdout as a valid JSON string wrapped in special tags: <SOLVER_OUTPUT> { ... } </SOLVER_OUTPUT>.

### Code Design Blueprint:
Your generated python script must follow this structure:

import json
from ortools.linear_solver import pywraplp

def solve():
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        print(json.dumps({"status": "ERROR", "message": "Solver init failed"}))
        return
    
    # ... define variables, constraints, objective ...
    
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        result = {
            "status": "OPTIMAL",
            "objective_value": solver.Objective().Value(),
        }
    else:
        result = {"status": "INFEASIBLE_OR_UNBOUNDED"}
        
    print(f"<SOLVER_OUTPUT>{json.dumps(result)}</SOLVER_OUTPUT>")

if __name__ == '__main__':
    solve()
"""

def call_gemini(prompt: str, system_inst: str) -> str:
    """Calls Gemini across your specifically active developer models."""
    models_to_try = [
        'gemini-3.5-flash',
        'gemini-2.5-pro',
        'gemini-2.5-flash'
    ]
    last_err = None
    
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_inst,
                    temperature=0.1,
                ),
            )
            return response.text
        except Exception as e:
            last_err = e
            continue
            
    raise RuntimeError(f"All API model endpoints failed. Last error: {last_err}")

def extract_code(text: str) -> str:
    """Robust extraction of python code from markdown fences."""
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def extract_solver_output(stdout: str) -> dict:
    """Robust extraction of solver JSON output."""
    match = re.search(r"<SOLVER_OUTPUT>\s*(.*?)\s*</SOLVER_OUTPUT>", stdout, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    return {"status": "PARSING_ERROR", "error": f"Invalid stdout received: {stdout}"}

def generate_solver_code(user_prompt: str, correction_context: str = "") -> str:  
    prompt = f"User Request: {user_prompt}\n"  
    if correction_context:  
        prompt += f"\nPrevious attempt failed with execution error:\n{correction_context}\nPlease debug the error and regenerate the corrected python code."

    raw_text = call_gemini(prompt, SYSTEM_INSTRUCTION)
    return extract_code(raw_text)

def execute_solver_code(code: str) -> dict:  
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:  
        temp_file.write(code.encode('utf-8'))  
        temp_path = temp_file.name

    try: 
        result = subprocess.run( 
            [sys.executable, temp_path], 
            capture_output=True, 
            text=True, 
            timeout=15 
        ) 
        os.unlink(temp_path)
        
        if result.returncode != 0: 
            return {"status": "CRASHED", "error": result.stderr} 
            
        return extract_solver_output(result.stdout)
    except subprocess.TimeoutExpired: 
        if os.path.exists(temp_path):
            os.unlink(temp_path) 
        return {"status": "TIMEOUT", "error": "Solver execution timed out."}

def run_agent_loop(user_input: str):  
    print("\n[OptiMind] 🧠 Analyzing business problem and dynamically generating optimization model...")

    correction_msg = "" 
    max_attempts = 3 
    
    for attempt in range(1, max_attempts + 1): 
        print(f" -> Code Generation (Attempt {attempt}/{max_attempts})...") 
        generated_code = generate_solver_code(user_input, correction_msg) 
        
        print(" -> Executing solver engine securely...") 
        outcome = execute_solver_code(generated_code) 
        
        if outcome.get("status") in ["CRASHED", "PARSING_ERROR"]: 
            print(f" ⚠️ Code execution failed. Triggering automated self-correction...") 
            correction_msg = outcome.get("error", "Unknown execution error.") 
        else: 
            print(" ✅ Optimization successfully executed!") 
            
            summary_prompt = f""" 
            User original problem: {user_input} 
            Optimal Solver JSON result: {json.dumps(outcome)} 
            
            Synthesize a highly polished, professional summary explaining the optimal plan and why it is the most efficient choice. 
            Highlight key metrics such as minimized cost, allocation flows, and constraint satisfaction. 
            """ 
            summary_text = call_gemini(
                summary_prompt,
                "You are a professional decision intelligence consultant. Present complex mathematical outcomes as clear, actionable, executive-friendly recommendations."
            )
            print("\n" + "="*50) 
            print("OPTIMIND DECISION SUMMARY") 
            print("="*50) 
            print(summary_text) 
            print("="*50) 
            return 
            
    print("\n❌ Failed to solve the problem after maximum correction iterations.")

if __name__ == '__main__':  
    if len(sys.argv) > 1:  
        query = " ".join(sys.argv[1:])  
    else:  
        query = ("We have supply plants in Mumbai (Capacity: 500 units, production cost: $10/unit, quality: 95%) "  
                 "and Pune (Capacity: 300 units, production cost: $12/unit, quality: 98%). "  
                 "We need to ship to our warehouse in Delhi which has a demand of 400 units. "  
                 "The average quality of shipping products delivered to Delhi must be at least 96%. "  
                 "The shipping cost from Mumbai to Delhi is $2/unit, and from Pune to Delhi is $1.5/unit. "  
                 "Find the lowest cost allocation of product shipping.")

    run_agent_loop(query)
