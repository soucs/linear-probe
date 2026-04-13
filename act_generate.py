from llm_generator import LLMGenerator
import json

model_path = "Qwen3.5-2B-4bit"
llm = LLMGenerator(model_path)

input_file = "data/deceptive_prompts_transformed.jsonl"
output_file = "data/deceptive_prompts_output.jsonl"

# Process the file
SAMPLES_TO_PROCESS = 50
with open(input_file, "r", encoding="utf-8") as f_in, \
    open(output_file, "w", encoding="utf-8") as f_out:
    
    i = 0 # To stop at 100 entries
    for line in f_in:
        if not line.strip():
            continue
            
        try:
            data = json.loads(line)
            output_text, act_array = llm.layer_wise_loop_generate(
                messages=data['inputs'],
                save_name=data['id'],
            )
            if output_text:
                data['output'] = output_text
                data['act_shape'] = act_array.shape
                f_out.write(json.dumps(data) + "\n")
                i += 1
                if i >= SAMPLES_TO_PROCESS:
                    break
        except Exception as e:
            print(f"Error processing line: {e}")

print(f"Done! Created {output_file}")