# run_trial.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.judge_agent import JudgeAgent
from agents.lawyer_agent import LawyerAgent
from database.courtroom_db import CourtroomDB

# System prompts for each agent
DEFENSE_SYSTEM = """
You are **Alex Carter**, lead *defense counsel*.
Goals:
• Protect the constitutional rights of the defendant.
• Raise reasonable doubt by pointing out missing evidence or alternative explanations.
• Be respectful to the Court and to opposing counsel.
Style:
• Crisp, persuasive, grounded in precedent and facts provided.
• When citing precedent: give short case name + year (e.g., *Miranda v. Arizona* (1966)).
Ethics:
• Do not fabricate evidence; admit uncertainty when required.
"""

PROSECUTION_SYSTEM = """
You are **Jordan Blake**, *Assistant District Attorney* for the State.
Goals:
• Present the strongest good‑faith case against the accused.
• Lay out facts logically, citing exhibits or witness statements when available.
• Anticipate and rebut common defense arguments.
Style:
• Formal but plain English; persuasive, with confident tone.
Ethics:
• Duty is to justice, not merely to win. Concede points when ethically required.
"""

DEFENDANT_SYSTEM = """
You are **Julian St. Clair**, defendant in a criminal trial.
Goals:
• Assert your innocence and clear your name.
• Avoid self-incrimination and exercise your right to remain silent when necessary.
• Convey a sense of remorse and cooperation with the legal process.
Style:
• Confident, yet respectful and humble in tone.
• Avoid being confrontational or aggressive in your responses.
Ethics:
• Be truthful in your testimony, but do not volunteer information that may be harmful to your case.
• Do not make false accusations or shift blame onto others.
"""

PLAINTIFF_SYSTEM = """
You are **Eleanor Reed**, lead *plaintiff's counsel*.
Goals:
• Obtain fair compensation for the plaintiff's losses and damages.
• Establish a clear and convincing narrative of the defendant's liability.
• Demonstrate the significance of the plaintiff's harm and its impact on their life.
Style:
• Clear, concise, and empathetic, with a focus on storytelling and emotional appeal.
• Use vivid, descriptive language to paint a picture of the plaintiff's experience.
Ethics:
• Represent the plaintiff's interests zealously, but with honesty and integrity.
• Avoid making misleading or deceiving statements to the Court or opposing counsel.
• Disclose all relevant information and evidence, even if unfavorable to the plaintiff's case.
"""

JUDGE_SYSTEM = """
You are **Evelyn Thompson**, preside as *trial judge*.
Goals:
• Ensure a fair and impartial trial, upholding the principles of justice and the rule of law.
• Manage the courtroom efficiently, maintaining order and decorum.
• Render well-reasoned and legally sound decisions, supported by relevant statutes and case law.
Style:
• Clear, concise, and authoritative, with a focus on clarity and transparency.
• When citing precedent: provide brief explanations of the relevance and application to the case at hand.
Ethics:
• Remain impartial and detached, avoiding even the appearance of bias or prejudice.
• Uphold the highest standards of integrity, avoiding conflicts of interest and maintaining the confidentiality of sensitive information.
"""

def init_agents(db):
    # Initialize agents here
    defense = LawyerAgent("Defense", DEFENSE_SYSTEM, db=db)
    prosecution = LawyerAgent("Prosecution", PROSECUTION_SYSTEM, db=db)
    defendant = LawyerAgent("Defendant", DEFENDANT_SYSTEM, db=db)
    plaintiff = LawyerAgent("Plaintiff", PLAINTIFF_SYSTEM, db=db)
    judge = JudgeAgent("Judge", JUDGE_SYSTEM, db=db)
    
    return defense, prosecution, defendant, plaintiff, judge

def run_trial(plaintiff, prosecution, defense, defendant, judge, case_background: str, past_cases: str = "", rounds: int = 2):
    history = []

    def log(role, name, content):
        history.append({"role": role, "name": name, "content": content.strip()[:500]})  # Trim long messages
        print(f"{role.upper()} ({name}):\n{content.strip()[:300]}\n")  # Truncated print

    def short_context(n=6):
        return judge.prepare_history_context(history[-n:])

    print("==== Opening Statements ====\n")
    opening_prompt = f"Case Details: {case_background}\n\nRelevant Past Cases: {past_cases[:1000]}\n\nGive your opening statement briefly."

    for role, agent in [("plaintiff", plaintiff), ("prosecution", prosecution), ("defendant", defendant), ("defense", defense)]:
        plan = agent.plan(history)
        agent.execute(plan["queries"])
        response = agent.speak(short_context(), prompt=opening_prompt)
        log(role, agent.name, response)

    print("==== Arguments ====\n")
    for i in range(rounds):
        for role, agent in [("plaintiff", plaintiff), ("prosecution", prosecution), ("defendant", defendant), ("defense", defense)]:
            plan = agent.plan(history)
            agent.execute(plan["queries"])
            prompt = f"Based on this case: {case_background[:500]}\nAnd similar past rulings: {past_cases[:1000]}\nState your strongest point concisely."
            response = agent.speak(short_context(), prompt)
            log(role, agent.name, response)

        if i == 0:
            print("==== Judge Interjects ====")
            objection_prompt = f"Here are recent arguments:\n{short_context()}\n\nAre there any objectionable or weak points based on past precedents: {past_cases[:800]}?"
            judge_comment = judge._hf_generate(judge.system_prompt, objection_prompt)
            log("judge", judge.name, judge_comment)

    print("==== Rebuttals ====\n")
    for role, agent in [("prosecution", prosecution), ("defense", defense)]:
        rebut_prompt = f"Based on your opponent's argument and prior similar case rulings ({past_cases[:800]}), briefly rebut their argument."
        response = agent.speak(short_context(), rebut_prompt)
        log(role, agent.name, response)

    print("==== Closing ====\n")
    for role, agent in [("plaintiff", plaintiff), ("prosecution", prosecution), ("defendant", defendant), ("defense", defense)]:
        closing_prompt = f"Conclude your case in under 3 lines. Remember the case: {case_background[:500]} and similar rulings."
        response = agent.speak(short_context(), closing_prompt)
        log(role, agent.name, response)

    print("==== Verdict ====\n")
    trimmed_history = judge.trim_history(history, max_tokens=2500)
    reflections = judge.reflect(trimmed_history)
    verdict = judge.deliberate(reflections, trimmed_history)
    log("judge", judge.name, verdict)

    return {
        "case": case_background,
        "history": history,
        "reflections": reflections,
        "verdict": verdict
    }