from openai import OpenAI
import random
import csv
from collections import Counter
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """

You are an expert linguistic annotator.  
Your task is to annotate pairs of words for six generative ordering constraints:  
Form, Percept, Culture, Power, Intense, and Icon.

Each constraint is binary-coded from the perspective of the *first* word in the pair:

• 1  = first word favors the constraint  
• -1 = second word favors the constraint  
• 0  = neither word favors the constraint  

Your output must ALWAYS be exactly six numbers separated by spaces, 
in this fixed order:
Form Percept Culture Power Intense Icon

No explanations, no text, no punctuation, no lists.  
Only six numbers.

------------------------------------------------------------
CONSTRAINT DEFINITIONS
(optimized for fast, deterministic application)
------------------------------------------------------------

FORM (Relative Formal Markedness)
The word with the more general, less marked meaning appears earlier.
Less marked = broader meaning, more general category, simpler, 
less dependent on context, or the superset in a logical subset relation.
Examples:
• flowers and roses → flowers less marked → 1
• boards and two-by-fours → boards less marked → 1
• changing and improving → changing less marked → 1
Code:
1  = first word is less marked / more general  
-1 = second word is  
0  = neither clearly is  

PERCEPT (Perceptual Markedness)
Elements more closely connected to the speaker appear first.
Less marked = more animate, more concrete, more proximal,
positive, singular, here/front/above, etc.
Examples:
• people and soils → animate before inanimate → 1
• action and mind → concrete before abstract → 1
Code based on which word is less perceptually marked:
1, -1, or 0.

CULTURE (Cultural Centrality in American culture)
The more culturally central, common, salient concept appears first.
Examples:
• oranges and grapefruit → oranges more central → 1
• mother and dad → mother more central to caregiving → 1
Code:
1, -1, or 0.

POWER (Social/Cultural Power or Priority)
The more powerful, authoritative, serious-consequence, or dominant
concept appears first.
Examples:
• clergymen and parishioners → clergy more powerful → 1
• laws and rules → laws carry greater consequences → 1
Code: 1, -1, or 0.

INTENSE (Intensity / Extremeness)
The more intense, stronger, or more extreme concept appears first.
Examples:
• war and peace → war more intense → 1
• cruel and unusual → cruel more intense → 1
Code: 1, -1, or 0.

ICON (Sequential / Prerequisite / Scalar Ordering)

A non-zero code is ONLY used if the two words inherently *require* or *encode*
a BEFORE-AFTER, prerequisite, causal, or scalar order in their meaning.

Valid examples:
• achieve → maintain
• slow → stop
• before → after
• here → there (movement)
• cause → effect

NOT valid:
• historical evolution (e.g., radios → televisions)
• technological progression, invention timeline
• cultural development
• general improvement or upgrading
• popularity or frequency
• newer vs older devices
• “used after” in typical usage, unless required by meaning

If the pair does NOT encode a necessary sequence, always 0.


------------------------------------------------------------
RESPONSE FORMAT
------------------------------------------------------------
Your response for “WordA WordB” must be:

Form Percept Culture Power Intense Icon

For example:
Input: agencies individuals  
Output: 0 -1 0 0 0 0

No other text. Ever.

------------------------------------------------------------
FEW-SHOT EXAMPLES
(Do NOT explain them; just learn the pattern.)
------------------------------------------------------------

20s 30s                0  0  1  0  0  1
a.b. m.a.              0  0  0 -1  0  1
abasement humiliation -1  0  0  0  1  1
ability age            0 -1  0  0  0 -1
ability desire         0  1 -1  0  0  0
ability strength       0 -1  0  0  0  0
abolition emancipation 1  0  0  0  0  0
action character       0 -1  0  0  0  0
action conversation    0  0  0  0  0  0
action mind            0  1  0  0  0  0
action motion         -1  0  0  0  0  0
actions feelings       0  1  0  0  0  0
activities character   0  1  0  0  0  0
activities places      0  0  0  0  0  0
activity nature        0  1  0  0  0  0
addresses names        0  0 -1  0  0  0
adventure romance      0  0  0  0  0  0
africa asia            0  0 -1 -1  0  0
age day                0  0  0  0  0 -1
agencies individuals   0 -1  0  0  0  0


"""

def annotate_pair(model, system_prompt, wordA, wordB, N=10):
    print(f"Processing: {wordA}, {wordB}")
    """
    Returns the modal annotations for a single pair (wordA, wordB).
    """
    responses = []
    user_prompt = f"{wordA}, {wordB}"
    for _ in range(N):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = resp.choices[0].message.content.strip()
        print(content)
        # Expect something like: "0 -1 0 0 0 0"
        numbers = list(map(int, content.split()))
        responses.append(numbers)
    # responses is N × 6
    # We now take the mode for each column
    modal = []
    for i in range(6):
        column = [r[i] for r in responses]
        most_common = Counter(column).most_common(1)[0][0]
        modal.append(most_common)
    return modal  # [Form, Percept, Culture, Power, Intense, Icon]





# response = client.chat.completions.create(
#     model="gpt-5-nano",
#     messages=[
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": "church, graveyard"}
#     ]
# )

#print(response.choices[0].message.content)


N = 2 #how many times to sample each word pair

test = annotate_pair(
    model = 'gpt-5-nano',
    system_prompt = system_prompt,
    wordA='buy',
    wordB= 'sell',
    N = 5
)

print(test)
