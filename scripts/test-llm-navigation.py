#!/usr/bin/env python3
"""Test LLM-based agent navigation in SimWorld.

Uses Claude's vision to analyze ego-view images and decide navigation
actions. The agent sees the 3D scene and reasons about how to reach
a target location described in natural language.

Usage:
    python3 scripts/test-llm-navigation.py --host macpro51 --port 9000
    python3 scripts/test-llm-navigation.py --host macpro51 --provider ollama
"""
import argparse
import base64
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/Volumes/SSDRAID0/agentic-system/intelligent-agents')

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


SYSTEM_PROMPT = """\
You are a navigation agent. You must reach a target position.

RULES:
1. If relative bearing says "left" → turn_left
2. If relative bearing says "right" → turn_right
3. If relative bearing says "ahead" → move_forward
4. If bearing magnitude > 60° → turn 90°
5. If bearing magnitude 15-60° → turn 45°
6. If bearing < 15° → move_forward 400

Available actions:
  move_forward 200
  move_forward 400
  turn_left 45
  turn_right 45
  turn_left 90
  turn_right 90

Reply with ONLY the action string. No explanation."""


def create_vision_prompt(pos, yaw, target, distance, step):
    """Build the text prompt that accompanies the ego-view image."""
    # Compute bearing to target
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    bearing = math.degrees(math.atan2(dy, dx))
    relative = ((bearing - yaw + 180) % 360) - 180

    if abs(relative) < 15:
        direction = "directly ahead"
    elif abs(relative) < 60:
        direction = "slightly to the right" if relative > 0 else "slightly to the left"
    elif abs(relative) < 120:
        direction = "to the right" if relative > 0 else "to the left"
    else:
        direction = "behind you"

    return (
        f"Step {step}. Distance to target: {distance:.0f}. "
        f"Target is {direction} (relative bearing {relative:.0f}°).\n"
        f"Action:"
    )


def llm_navigation_agent_claude(image_path, pos, yaw, target, distance, step):
    """Use Claude vision to decide the next action."""
    client = anthropic.Anthropic()

    with open(image_path, 'rb') as f:
        image_data = f.read()
    image_b64 = base64.b64encode(image_data).decode('utf-8')

    prompt = create_vision_prompt(pos, yaw, target, distance, step)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    action = message.content[0].text.strip()
    # Extract just the action if the model added extra text
    for line in action.split('\n'):
        line = line.strip()
        if line.startswith(('move_forward', 'turn_left', 'turn_right')):
            return line
    return action


def _resize_image_b64(image_path, max_size=320):
    """Resize image to reduce vision model processing time."""
    from PIL import Image
    import io
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=80)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def llm_navigation_agent_ollama(image_path, pos, yaw, target, distance, step):
    """Use local Ollama vision model to decide the next action."""
    import httpx

    image_b64 = _resize_image_b64(image_path)

    prompt = create_vision_prompt(pos, yaw, target, distance, step)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    ollama_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    model = os.environ.get('OLLAMA_VISION_MODEL', 'llava-llama3:8b-v1.1-fp16')
    resp = httpx.post(
        f'{ollama_url}/api/generate',
        json={
            'model': model,
            'prompt': full_prompt,
            'images': [image_b64],
            'stream': False,
            'options': {'num_predict': 50, 'temperature': 0.1},
        },
        timeout=180,
    )
    resp.raise_for_status()
    action = resp.json()['response'].strip()
    # Fix markdown-escaped underscores from some models
    action = action.replace('\\_', '_')

    for line in action.split('\n'):
        line = line.strip()
        if line.startswith(('move_forward', 'turn_left', 'turn_right')):
            return line
    return action


def llm_navigation_agent_openai(image_path, pos, yaw, target, distance, step):
    """Use OpenAI GPT-4o vision to decide the next action."""
    client = openai.OpenAI()

    with open(image_path, 'rb') as f:
        image_data = f.read()
    image_b64 = base64.b64encode(image_data).decode('utf-8')

    prompt = create_vision_prompt(pos, yaw, target, distance, step)

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=100,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                    "detail": "low",
                }},
                {"type": "text", "text": prompt},
            ]},
        ],
    )

    action = response.choices[0].message.content.strip()
    for line in action.split('\n'):
        line = line.strip()
        if line.startswith(('move_forward', 'turn_left', 'turn_right')):
            return line
    return action


def fallback_rule_agent(pos, yaw, target):
    """Rule-based fallback if LLM response is unparseable."""
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    distance = math.sqrt(dx * dx + dy * dy)
    desired_yaw = math.degrees(math.atan2(dy, dx))
    diff = (desired_yaw - yaw + 180) % 360 - 180

    if abs(diff) > 15:
        if diff > 0:
            return f'turn_right {min(abs(diff), 45)}'
        else:
            return f'turn_left {min(abs(diff), 45)}'
    return f'move_forward {min(200, distance)}'


def main():
    parser = argparse.ArgumentParser(description='LLM-based SimWorld navigation')
    parser.add_argument('--host', default='macpro51')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--provider', default='claude', choices=['claude', 'ollama', 'openai'])
    parser.add_argument('--max-steps', type=int, default=30)
    args = parser.parse_args()

    from prometheus.environments.simworld_env import (
        SimWorldConfig,
        SimWorldEnvironment,
    )

    config = SimWorldConfig(server_host=args.host, server_port=args.port)
    env = SimWorldEnvironment(config)

    print(f'Connecting to SimWorld at {args.host}:{args.port}...')
    env.connect()
    print(f'Connected! Using {args.provider} vision for navigation.\n')

    # Navigation task
    start = (300, 50, 100)
    target = (1700, -1700, 100)
    print(f'Task: Navigate from {start[:2]} to {target[:2]}')
    print(f'Success radius: 300 units\n')

    obs = env.reset(spawn_position=start)
    providers = {
        'claude': llm_navigation_agent_claude,
        'ollama': llm_navigation_agent_ollama,
        'openai': llm_navigation_agent_openai,
    }
    agent_fn = providers[args.provider]

    results = []
    for step in range(1, args.max_steps + 1):
        pos = env.get_position()
        dist = env.distance_to(target)
        yaw = env._yaw

        if dist < 300:
            print(f'\n  Target reached in {step - 1} steps!')
            break

        # Get LLM action from ego-view
        t0 = time.time()
        used_fallback = False
        try:
            action = agent_fn(obs.raw_image_path, pos, yaw, target, dist, step)
        except Exception as e:
            print(f'  LLM error: {e}')
            action = fallback_rule_agent(pos, yaw, target)
            used_fallback = True

        # Clean action: strip degree symbols, extra punctuation
        import re
        action = re.sub(r'[°º]', '', action).strip()

        # Validate action
        valid_prefixes = ('move_forward', 'turn_left', 'turn_right')
        if not action.startswith(valid_prefixes):
            print(f'  Invalid LLM action: {action!r} -> using fallback')
            action = fallback_rule_agent(pos, yaw, target)
            used_fallback = True

        # Stuck-loop detection: if same action 3 times in a row, use fallback
        if (len(results) >= 2
                and not used_fallback
                and results[-1]['action'] == action
                and results[-2]['action'] == action):
            action = fallback_rule_agent(pos, yaw, target)
            used_fallback = True

        dt = time.time() - t0
        obs, _, _ = env.step(action)

        new_pos = env.get_position()
        new_dist = env.distance_to(target)
        tag = '  [fallback]' if used_fallback else ''
        results.append({
            'step': step,
            'action': action,
            'pos': (float(new_pos[0]), float(new_pos[1])),
            'dist': float(new_dist),
            'llm_time': dt,
            'fallback': used_fallback,
        })

        print(f'  Step {step:3d}: {action + tag:30s} '
              f'pos=({new_pos[0]:.0f}, {new_pos[1]:.0f}) '
              f'dist={new_dist:.0f} '
              f'({dt:.1f}s)')
    else:
        print(f'\n  Did not reach target in {args.max_steps} steps.')
        print(f'  Final distance: {env.distance_to(target):.0f}')

    # Summary
    total_llm_time = sum(r['llm_time'] for r in results)
    fallbacks = sum(1 for r in results if r.get('fallback', False))
    print(f'\n=== Summary ===')
    print(f'  Steps: {len(results)}')
    print(f'  Final distance: {env.distance_to(target):.0f}')
    print(f'  LLM calls: {len(results) - fallbacks} ({args.provider})')
    print(f'  Fallbacks: {fallbacks}')
    print(f'  Total LLM time: {total_llm_time:.1f}s')
    print(f'  Avg per step: {total_llm_time / max(len(results), 1):.1f}s')

    if obs.raw_image_path and os.path.exists(obs.raw_image_path):
        print(f'  Final ego view: {obs.raw_image_path}')

    env.disconnect()
    print('\nDone!')


if __name__ == '__main__':
    main()
