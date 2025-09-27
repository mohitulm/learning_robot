# llama_processor.py — HF Transformers only, no Unsloth, no gated models

import os, re, torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from transformers import AutoTokenizer, AutoModelForCausalLM

# You can override with env var: HF_MODEL or HF_MODELS (comma-separated)
DEFAULT_MODELS = [
    # tiny + open models first (fast on CPU)
    "Qwen/Qwen2.5-0.5B-Instruct",
    #"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # still open but bigger (may be slow on CPU; fine on small GPU)
    #"Qwen/Qwen2.5-1.5B-Instruct",
    #"microsoft/Phi-3-mini-4k-instruct",  # ~3B
]

import re

PATTERN = re.compile(r"""
    ACTION          # literal
    \s*:? \s*       # optional colon
    ([a-z][\w-]*)   # action: any token (letters/digits/_/-)
    \s*[,;]? \s*    # optional comma/semicolon
    TARGET
    \s*:? \s*
    ([a-z0-9_][\w-]*) # target: token or 'none'
""", re.IGNORECASE | re.VERBOSE)


class LlamaProcessor(Node):
    def __init__(self):
        super().__init__('llama_processor')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        models_to_try = []
        if os.getenv("HF_MODEL"):
            models_to_try = [os.getenv("HF_MODEL")]
        elif os.getenv("HF_MODELS"):
            models_to_try = [m.strip() for m in os.getenv("HF_MODELS").split(",") if m.strip()]
        else:
            models_to_try = DEFAULT_MODELS

        self.tokenizer, self.model, self.model_id_used = self._load_first_available(models_to_try)

        self.sub = self.create_subscription(String, 'voice_commands', self.process_command, 10)
        self.pub = self.create_publisher(String, 'llm_commands', 10)
        self.get_logger().info(f"Backend: HF Transformers on {self.device.upper()} | Model: {self.model_id_used}")

    def _load_first_available(self, model_ids):
        last_err = None
        for mid in model_ids:
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                if tok.pad_token_id is None:
                    tok.pad_token = tok.eos_token
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                mdl = AutoModelForCausalLM.from_pretrained(
                    mid, torch_dtype=dtype, low_cpu_mem_usage=True
                ).to(self.device)
                return tok, mdl, mid
            except Exception as e:
                self.get_logger().warn(f"Model load failed for {mid}: {e}")
                last_err = e
        raise RuntimeError(f"No model could be loaded. Last error: {last_err}")

    def _messages(self, user_input: str):
        sys = (""" Parse a command to output exactly in the format: ACTION <move|stop> TARGET <location|none>
Lowercase, four words, single spaces, no punctuation or extra text. 
Examples: 
            - "Go to the kitchen" → ACTION: move, TARGET: kitchen
            - "Stop moving" → ACTION: stop, TARGET: none
"""
            
        )
        return [{"role": "system", "content": sys},
                {"role": "user", "content": user_input}]

    def _generate(self, messages):
        # Build text then tokenize to ensure we pass attention_mask
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"System:\n{messages[0]['content']}\nUser:\n{messages[1]['content']}\nAssistant:\n"

        enc = self.tokenizer(
            text, return_tensors="pt", return_attention_mask=True, padding=False
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _parse(self, text: str):
        response = text.split('\n')[-1]
        m = PATTERN.search(response)
        if not m:
            return None
        action = m.group(1).lower().strip()
        target = re.sub(r"[^\w \-]+$", "", m.group(2).strip()).lower()
        if target in ("none", "null"):
            target = ""
        return {"action": action, "target": target}

    def process_command(self, msg):
        user_input = msg.data if isinstance(msg, String) else str(msg)
        self.get_logger().info(f"Processing command: {user_input}")
        text = self._generate(self._messages(user_input))
        parsed = self._parse(text)
        if parsed:
            payload = f"{parsed['action']}|{parsed['target']}"
            self.pub.publish(String(data=payload))
            self.get_logger().info(f"Published: {payload}")
        else:
            self.get_logger().warn(f"LLM response could not be parsed: {text!r}")

def main(args=None):
    rclpy.init(args=args)
    node = LlamaProcessor()
    #text_input = input('Give a command : ')
    #node.process_command(text_input)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
