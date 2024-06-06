import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练的GPT-2模型和tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 设置Streamlit页面标题
st.title("ChatGPT UI")

# 添加文本框用于用户输入
user_input = st.text_input("输入你的消息:", "")

# 如果用户输入了消息
if user_input:
    # 对用户输入的消息进行tokenize
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    # 使用模型生成回复
    reply_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=model.config.pad_token_id)
    # 解码模型生成的回复
    reply_text = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    # 在页面上显示模型生成的回复
    st.text_area("ChatGPT 的回复:", value=reply_text, height=200, max_chars=None, key=None)

# 注：这只是一个简单的示例，可能需要进一步改进以提高用户体验和模型的交互性
