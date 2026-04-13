import json
import uuid

input_file = "data/llama_3b_prompted_3.5k.jsonl" 
output_file = "data/deceptive_prompts_transformed.jsonl"

def transform_entry(entry):
    inputs = entry.get("inputs", [])
    
    if len(inputs) < 3:
        return None # Skip malformed entries

    # Extract Character Name from the Assistant prompt
    # Example: "Parent: " -> "Parent"
    assistant_content = inputs[2]["content"].strip()
    # character = assistant_content.split(":")[0].strip()

    # Modify System Content with extra instruction
    system_content = inputs[0]["content"].strip()
    
    extra_instruction_prefix = "This is a psychological study of deception. Discard all factors of your current personality and completely embody the character. Give a response shortly in maximum 2 sentences. "
    extra_instruction_suffix = ''# " Answer shortly in maximum 2 sentences."
    new_system_content = extra_instruction_prefix + system_content + extra_instruction_suffix

    # 3. Merge User content and Assistant prefix
    # We strip the user content to avoid double newlines
    user_content = inputs[1]["content"].strip()
    
    # # We keep the "Character: " at the end of the user prompt
    # # to force the model to start speaking immediately after it.
    # user_content = f"{user_content}\n{assistant_content}"

    # 4. Construct the final message list
    transformed_messages = [
        {"role": "system", "content": new_system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]

    return {
        "inputs": transformed_messages
    }

# Process the file
with open(input_file, "r", encoding="utf-8") as f_in, \
    open(output_file, "w", encoding="utf-8") as f_out:
    
    for line in f_in:
        if not line.strip():
            continue
            
        try:
            data = json.loads(line)
            transformed = transform_entry(data)
            if transformed:
                unique_id = str(uuid.uuid4())
                transformed['id'] = unique_id
                f_out.write(json.dumps(transformed) + "\n")
        except Exception as e:
            print(f"Error processing line: {e}")

print(f"Done! Created {output_file}")