import asyncio
import os
import re
from openai import OpenAI

from models import AnigravityAction
from server.anigravity_env_environment import AnigravityEnvironment

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_key")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "easy_hover")
MAX_STEPS = 20

SYSTEM_PROMPT = """
You are an AI controlling an anti-gravity drone. 
Output ONLY a single number between 0.0 and 1.0 representing thrust power.
If Current < Target, output a number > 0.5 to go up.
If Current > Target, output a number < 0.5 to go down.
"""

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error="null"):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = AnigravityEnvironment()
    
    rewards = []
    steps_taken = 0
    
    log_start(task=TASK_NAME, env="anigravity-drone-stabilizer", model=MODEL_NAME)
    
    try:
        state = await env.reset()
        obs_dict = state.observation
        
        for step in range(1, MAX_STEPS + 1):
            altitude = obs_dict["altitude"]
            target = obs_dict["target_altitude"]
            velocity = obs_dict["velocity"]

            user_prompt = f"Target: {target} | Current: {altitude:.1f} | Velocity: {velocity:.1f}. Output thrust (0.0-1.0):"
            
            thrust = 0.5 
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=5,
                )
                reply = completion.choices[0].message.content.strip()
                match = re.search(r"0\.\d+|1\.0|0\.0|1|0", reply)
                if match:
                    thrust = float(match.group())
            except Exception:
                # Smart Fallback if API fails
                thrust = 0.8 if altitude < target else 0.2
                
            action = AnigravityAction(thrust_level=thrust)
            state = await env.step(action)
            obs_dict = state.observation
            reward = state.reward
            done = state.done
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=f"thrust({thrust})", reward=reward, done=done)
            
            if done:
                break
                
        max_possible = steps_taken * 1.0
        score = sum(rewards) / max_possible if max_possible > 0 else 0.0
        success = score > 0.5
        
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        
    except Exception as e:
        print(f"[DEBUG] Error: {e}")

if __name__ == '__main__':
    asyncio.run(main())