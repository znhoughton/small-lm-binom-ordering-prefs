import pandas as pd
import logging
import random
from openai import OpenAI
import os
from tqdm import tqdm 
#so easy to do with Chat GPT

# ---------- SETUP ----------
client = OpenAI() 
MODEL = "gpt-5"  

logging.basicConfig(
    filename="gpt5_binomial_prompting.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------- PROMPT FUNCTION ----------
def ask_model(alpha: str, nonalpha: str) -> int:
    """
    Prompt the model to choose the more natural ordering.
    Returns 1 if Alpha chosen, 0 if Nonalpha chosen.
    Counterbalances whether Alpha is shown first or second.
    """
    # Randomize whether alpha or nonalpha is A
    if random.random() < 0.5:
        # Alpha is A, Nonalpha is B
        option_map = {"A": "Alpha", "B": "Nonalpha"}
        prompt = f"""
        Between these two phrases, which sounds more natural in English?
        A: {alpha}
        B: {nonalpha}

        **You must answer exactly with either 'A' or 'B'. Do not write anything else.**
        """
    else:
        # Nonalpha is A, Alpha is B
        option_map = {"A": "Nonalpha", "B": "Alpha"}
        prompt = f"""
        Between these two phrases, which sounds more natural in English?
        A: {nonalpha}
        B: {alpha}

        **You must answer exactly with either 'A' or 'B'. Do not write anything else.**
        """

    system_prompt = """
    You are a brilliant linguist that knows more than any human. 

    You answer questions about languages.

    Answer only with "A" or "B". You use no other words.
    """


    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "low"},
        instructions=system_prompt,
        input=prompt,
    )
    #print(f"Message is: {prompt}") # for debugging
    #print(f"Response is: {response}") # for debugging
    choice = response.output_text.strip()
    #print(f"Choice variable is: {choice}") #for debugging

    if choice in option_map:
        return 1 if option_map[choice] == "Alpha" else 0
    else:
        logging.warning(f"Unexpected output: {choice}")
        return None

# ---------- MAIN ----------
def main():
    df = pd.read_csv("../Data/nonce_and_attested_binoms.csv")
    #df = df.head(2)

    judgements = []

    print_every_row = 25
    print_every_trial = 5

    for idx, row in tqdm(df.iterrows(), total = len(df), desc="Processing binomials"): #progress bar 
        alpha = row["Alpha"]
        nonalpha = row["Nonalpha"]

        results = []
        for i in range(10):  # ask 10 times
            result = ask_model(alpha, nonalpha)
            results.append(result)

            if idx % print_every_row == 0 and i % print_every_trial == 0:
                print(f"\nRow {idx}, Trial {i}:")
                print(f"Alpha: {alpha}, Nonalpha: {nonalpha}")
                print(f"Result: {result}")


            logging.info(f"Row {idx}, Trial {i}: Alpha='{alpha}', Nonalpha='{nonalpha}', Result={result}")

        judgements.append(results)

    df["Judgements"] = judgements
    # Optional: summary column with proportion of Alpha choices
    df["Alpha_Proportion"] = df["Judgements"].apply(lambda x: sum(v == 1 for v in x) / len(x))

    df.to_csv(f"{MODEL}_binomials_with_judgements.csv", index=False)
    logging.info("Finished processing. Saved to binomials_with_judgements.csv")

if __name__ == "__main__":
    main()
