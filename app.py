import json
import logging

import gradio as gr
import pandas as pd
import pytesseract
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from color_formatter import ColorFormatter
from question_db_helper import QuestionDBHelper
from question_details import QuestionDetails

# 创建一个logger
logger = logging.getLogger('ColorLogger')
logger.setLevel(logging.DEBUG)

# 创建一个控制台handler并设置级别为DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# 创建一个格式化器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
console_handler.setFormatter(ColorFormatter())

# 将控制台handler添加到logger
logger.addHandler(console_handler)

model_name = 'qwen2:7b'
chat_model = ChatOllama(
    base_url="http://localhost:11434",
    temperature=0,
    model=model_name
)


def identifiy_question(question_image):
    """
    错题识别：使用Google的pytesseract从错题图像中提取错题文本。

    参数:
    image (PIL.Image): 输入的错题图像。

    返回:
    question_text: 从图像中提取的错题文本。
    """
    logger.info("智能体开始识别题目... ...")
    question_text = pytesseract.image_to_string(question_image, lang='chi_sim+eng')
    logger.debug("从图像中提取的题目文本如下：\n=============\n" + question_text + "\n=============\n")
    logger.info("智能体结束识别题目... ...")
    return question_text


def classify_question(question_text):
    logger.info("智能体开始分类题目... ...")
    classify_question_prompt_template = """
    你是一个试卷问题智能助手，现在需要你判断下面的这个问题： {question} 属于什么学科。
    学科的范围为: ["语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "其它"]，
    你只需要回答2个字的学科范围中的名称即可。如果问题的文本内容都是英文，那学科类型就是"英语"。你不知道答案，就回答"未知"，不要试图编造答案。
    答案要包括：
        text -- 问题的文本内容
        question_type -- 问题的类型（例如，选择题、填空题）
        difficulty -- 问题的难度级别
        answers -- 问题的正确答案
        subject -- 问题所属的科目或主题
    最终的输出格式严格控制为Json格式，Json字符串外面不能有其它内容。
    """
    classify_question_prompt = ChatPromptTemplate.from_template(classify_question_prompt_template)
    classify_question_chain = (
            {"question": RunnablePassthrough()}
            | classify_question_prompt
            | chat_model
            | StrOutputParser()
    )
    question_detail = QuestionDetails
    result = classify_question_chain.invoke(question_text)
    logger.debug("题目分类信息：\n=============\n" + result + "\n=============\n")
    data = json.loads(result)
    for key, value in data.items():
        setattr(question_detail, key, value)

    logger.info("智能体结束分类题目... ...")
    return question_detail


def store_question(question_detail):
    answers = json.dumps(question_detail.answers, ensure_ascii=False)
    db_helper = QuestionDBHelper()
    # 插入一条新问题
    db_helper.insert_question(
        text=question_detail.text,
        question_type=question_detail.question_type,
        difficulty=question_detail.difficulty,
        answers=answers,
        subject=question_detail.subject
    )


def store_question(question_detail):
    logger.info("智能体开始储存题目... ...")
    answers = json.dumps(question_detail.answers, ensure_ascii=False)
    db_helper = QuestionDBHelper()
    # 插入一条新问题
    db_helper.insert_question(
        text=question_detail.text,
        question_type=question_detail.question_type,
        difficulty=question_detail.difficulty,
        answers=answers,
        subject=question_detail.subject
    )
    logger.info("智能体结束储存题目... ...")


def review_question(subject):
    logger.info("智能体开始回顾错题... ...")
    db_helper = QuestionDBHelper()
    questions = db_helper.get_questions_by_subject(subject)
    # 将查询结果转换为DataFrame
    columns = ["序号", "问题", "类型", "难度", "选项", "学科"]
    data = pd.DataFrame(questions, columns=columns)

    logger.info("智能体结束回顾错题... ...")
    return pd.DataFrame(data)


def enhance_question(selected_index: gr.SelectData, dataframe_origin):
    logger.info("智能体开始错题巩固... ...")
    # return f"{selected_index.row_value}."
    err_question_text = selected_index.row_value
    enhance_model = ChatOllama(
        base_url="http://localhost:11434",
        temperature=0.8,
        model=model_name
    )
    before_rag_template = "请参加这道题目（ {err_question_text} )的内容， 出1道同类的题目.如果你不知道答案，就回答不知道，不要试图编造答案。"
    before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
    after_rag_chain = (
            {"err_question_text": RunnablePassthrough()}
            | before_rag_prompt
            | enhance_model
            | StrOutputParser()
    )
    logger.info("智能体结束错题巩固... ...")

    return after_rag_chain.invoke(err_question_text)


def process_question(question_image, query):
    """
    处理上传的图像，提取其中的文字，并将其向量化。

    参数:
    image (PIL.Image): 上传的图像。
    query (str): 向助手提出的需求。

    返回:
    str: 成功消息。
    """
    logger.info("智能体开始执行指令... ...")
    # 从错题图像中提取错题文本
    question_text = identifiy_question(question_image)
    question_detail = classify_question(question_text)
    store_question(question_detail)

    before_rag_template = "根据下面的上下文（" + question_text + ")内容用中文回答问题: {query}.如果你不知道答案，就回答不知道，不要试图编造答案。"
    before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
    after_rag_chain = (
            {"query": RunnablePassthrough()}
            | before_rag_prompt
            | chat_model
            | StrOutputParser()
    )
    return after_rag_chain.invoke(query)


def main():
    """
    创建并启动Gradio应用。
    """
    index_page = gr.Interface(
        fn=process_question,
        inputs=[
            gr.Image(label="错题图片", height=250),
            gr.Textbox(value="请给出问题的描述及正确选项，并给出解题思路", placeholder="需要智能体做什么...",
                       label="指令")
        ],
        outputs=gr.Textbox(label="执行结果", lines=14),
        title="错题整理智能体",
        description="上传一张错题的图片，并向智能体发出指令，智能体会根据图片内容和指令要求给出答案。",
        examples=[
            ["examples/question1.png", "请给出问题的描述及正确选项，并给出解题思路"],
            ["examples/question2.png", "请给出问题的描述及正确选项，并给出解题思路"],
            ["examples/question3.png", "请给出问题的描述及正确选项，并给出解题思路"],
            ["examples/question4.png", "请给出问题的描述及正确选项，并给出解题思路"],
        ],
        allow_flagging="never",
        clear_btn="重来",
        submit_btn="启动",
    )

    with gr.Blocks() as his_page:
        gr.Markdown(
            """
            # 错题回顾智能体
            选择科目，智能体会根据你选择的科目，从数据库中读取错题，并展示错题列表。
            """)
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    input_dropdown = gr.Dropdown(
                        ["语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "其它"], label="请选择科目")
                    review_button = gr.Button("错题回顾")
                with gr.Column():
                    output_textbox = gr.Textbox(label="错题巩固"),

            gr.Markdown("点击错题列表，智能体将自动给出与选中题目知识点相近的题目，帮助你巩固同类题型。")
            with gr.Row():
                output_df = gr.Dataframe(
                    headers=["序号", "问题", "类型", "难度", "选项", "学科"],
                    type="pandas",
                    column_widths=[50, 300, 100, 100, 300, 100],
                    wrap=True,
                    label="错题列表"
                )
            review_button.click(fn=review_question, inputs=input_dropdown, outputs=output_df)
            output_df.select(fn=enhance_question, inputs=output_df, outputs=output_textbox)

    his_page.description = "选择科目，智能体会根据你选择的科目，从数据库中读取错题，并展示错题列表。"
    tabbed_interface = gr.TabbedInterface([index_page, his_page], ["错题解析", "错题回顾"])
    tabbed_interface.launch(share=True)


# 运行测试用例
if __name__ == "__main__":
    main()
