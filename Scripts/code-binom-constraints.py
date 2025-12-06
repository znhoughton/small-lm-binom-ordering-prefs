import pandas as pd
from collections import Counter
from tqdm import tqdm
from openai import OpenAI

# ----------------------------
# CONNECT TO VLLM ENDPOINT
# ----------------------------
client = OpenAI(
    base_url="https://gpt-oss-120b.vailsys.net/v1",
    api_key="EMPTY"   # vLLM usually doesn't require auth, but the client demands a key
)

# ----------------------------
# SYSTEM PROMPT (unchanged)
# ----------------------------
system_prompt = """

You are an expert linguistic annotator.

Your task is to annotate a pair of words for SIX generative ordering
constraints:

Form Percept Culture Power Intense Icon

For each constraint you MUST output one of:
1   = first word favors the constraint
-1  = second word favors the constraint
0   = neither word favors the constraint

Your output MUST be:
- exactly SIX numbers
- separated by single spaces
- in this order:
  Form Percept Culture Power Intense Icon
- with NO other text, punctuation, commentary, or formatting.

If you cannot decide, output 0 for that constraint.

============================================================
FAST DEFINITIONS (for deterministic coding)
============================================================

FORM (Relative Formal Markedness)
The LESS marked, MORE general, BROADER, or logically SUPERSET term comes first.
Use 1 if WordA is less marked; -1 if WordB is; 0 if unclear.

Common cues for "less marked":
- broader category
- simpler or default meaning
- not defined in relation to the other
- superordinate in a subset relation
Examples: flowers–roses → 1 ; boards–two-by-fours → 1

PERCEPT (Perceptual Markedness)
The element CLOSER TO SPEAKER experience comes first:
animate > inanimate, concrete > abstract, positive > negative,
singular > plural (count nouns), here/front/above > there/back/below, etc.
1 = WordA less perceptually marked; -1 WordB; 0 neither.

CULTURE (Cultural Centrality in American culture)
The more culturally central, common, prototypical, or salient concept comes first.
Examples: oranges–grapefruit → 1 ; mother–dad → 1
Code 1 / -1 / 0.

POWER (Social or institutional hierarchy, seriousness, authority)
Whichever concept has more power, authority, dominance, consequence, or status
comes first.
Examples: clergymen–parishioners → 1 ; laws–rules → 1
Code 1 / -1 / 0.

INTENSE (Intensity / Extremeness)
The stronger, more extreme, more forceful element comes first.
Examples: war–peace → 1 ; cruel–unusual → 1; rain-snow → 0;
Code 1 / -1 / 0.

ICON (Sequential / Prerequisite / Scalar Ordering)
NON-ZERO ONLY IF the pair *requires* a BEFORE-AFTER or prerequisite relation
inherent to meaning (chronological, causal, directional, scalar).

Valid: achieve→maintain, slow→stop, before→after, cause→effect, here→there (movement).
INVALID:
- invention history
- cultural/technological evolution
- newer/older devices
- mere typical order of usage
- popularity or frequency
If no inherent sequence → 0.

============================================================
CRITICAL RULES (OSS models must obey)
============================================================
• Respond ONLY with six numbers.  
• No explanation, no definitions, no restatement of input.  
• No commas, brackets, quotes, or labels.  
• If any constraint is ambiguous → output 0 for that constraint.  
• NEVER invent new constraints.  
• ALWAYS maintain output order:
  Form Percept Culture Power Intense Icon

============================================================
FEW-SHOT EXAMPLES (Learn the pattern. DO NOT EXPLAIN.)
============================================================
20s 30s                0 0 1 0 0 1
a.b. m.a.              0 0 0 -1 0 1
abasement humiliation  -1 0 0 0 1 1
ability age            0 -1 0 0 0 -1
ability desire         0 1 -1 0 0 0
ability strength       0 -1 0 0 0 0
abolition emancipation 1 0 0 0 0 0
action character       0 -1 0 0 0 0
action conversation    0 0 0 0 0 0
action mind            0 1 0 0 0 0
action motion         -1 0 0 0 0 0
actions feelings       0 1 0 0 0 0
activities character   0 1 0 0 0 0
activities places      0 0 0 0 0 0
activity nature        0 1 0 0 0 0
addresses names        0 0 -1 0 0 0
adventure romance      0 0 0 0 0 0
africa asia            0 0 -1 -1 0 0
age day                0 0 0 0 0 -1
agencies individuals   0 -1 0 0 0 0

============================================================
OUTPUT FORMAT REMINDER
============================================================
ALWAYS output:
Form Percept Culture Power Intense Icon

Example:
Input: agencies individuals
Output: 0 -1 0 0 0 0


"""

# ----------------------------
# ANNOTATION FUNCTION
# ----------------------------
def annotate_pair(model, system_prompt, wordA, wordB, N=10, debug=True):
    """
    Returns modal [Form, Percept, Culture, Power, Intense, Icon].
    Includes optional debugging output.
    """
    user_prompt = f"{wordA} {wordB}"
    responses = []

    print(f"\nProcessing: {wordA}, {wordB}")

    for i in range(N):

        if debug:
            print(f"\n--- Request {i+1}/{N} ---")
            print("User prompt:", user_prompt)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1000,
            temperature=0.0,
        )

        if debug:
            print("Raw response object:", resp)

        # ---- SAFETY CHECKS ----
        if not resp or not getattr(resp, "choices", None):
            print("⚠ ERROR: vLLM returned no choices.")
            print("  Full response:", resp)
            responses.append([0, 0, 0, 0, 0, 0])
            continue

        msg = resp.choices[0].message

        if msg is None or msg.content is None:
            print("⚠ ERROR: message.content is None!")
            print("  Full response:", resp)
            responses.append([0, 0, 0, 0, 0, 0])
            continue

        content = msg.content.strip()

        if debug:
            print("Model content:", repr(content))

        # ---- PARSE OUTPUT ----
        try:
            numbers = list(map(int, content.split()))
            if len(numbers) != 6:
                raise ValueError("Did not receive six numbers.")
        except Exception as e:
            print("⚠ WARNING: Could not parse the output:", content)
            print("  Error:", e)
            print("  Substituting zeros.")
            numbers = [0, 0, 0, 0, 0, 0]

        responses.append(numbers)

    # ---- COMPUTE MODAL VALUES ----
    modal = []
    for j in range(6):
        col = [r[j] for r in responses]
        modal_val = Counter(col).most_common(1)[0][0]
        modal.append(modal_val)

    return modal



# ----------------------------
# PIPELINE ON CSV
# ----------------------------
input_csv = "fintetuning-binoms-without-constraints-coded.csv"
output_csv = "binomials_coded.csv"

# IMPORTANT: vLLM's "model name" is the *path* you passed to --model/--system-model
model = "/models/gpt-oss-120b"   
N = 10

df = pd.read_csv(input_csv)
#df = df.head(10) #for debugging

assert "WordA" in df.columns and "WordB" in df.columns

form_vals = []
percept_vals = []
culture_vals = []
power_vals = []
intense_vals = []
icon_vals = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    wordA = str(row["WordA"])
    wordB = str(row["WordB"])

    F, P, C, Pw, I, Ic = annotate_pair(model, system_prompt, wordA, wordB, N=N, debug = False)

    form_vals.append(F)
    percept_vals.append(P)
    culture_vals.append(C)
    power_vals.append(Pw)
    intense_vals.append(I)
    icon_vals.append(Ic)

df["Form"]    = form_vals
df["Percept"] = percept_vals
df["Culture"] = culture_vals
df["Power"]   = power_vals
df["Intense"] = intense_vals
df["Icon"]    = icon_vals

df.to_csv(output_csv, index=False)
print(f"\n✨ Done! Saved annotated CSV to: {output_csv}")
