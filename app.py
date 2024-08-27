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
    """
    整理问题，根据问题文本的内容，判断问题的学科及其它问题信息

    参数:
    question_text (str): 用户提出的问题文本。

    返回:
    question_detail： 整理后的问题详情。

    """

    logger.info("智能体开始整理分类题目... ...")
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

    logger.info("智能体结束整理分类题目... ...")
    return question_detail


def store_question(question_detail):
    """
    将问题存储到数据库中。

    本函数负责将问题详情（包括问题文本、类型、难度、答案及科目）存储到数据库中。
    它首先将答案序列化为JSON格式的字符串，然后通过数据库助手插入问题。

    参数:
    - question_detail: 问题详情对象，包含问题的所有必要信息。

    返回值:
    无

    重要性说明:
    本函数对于持久化问题信息至关重要，确保问题及其相关答案能被正确存储，
    以便后续检索和使用。
    """
    logger.info("智能体开始储存题目... ...")
    # 将问题答案序列化为JSON格式的字符串，确保非ASCII字符不受影响
    answers = json.dumps(question_detail.answers, ensure_ascii=False)
    # 创建数据库助手实例，用于操作问题数据库
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
    """
    根据科目回顾错题。

    此函数的目的是从数据库中检索指定科目下的所有错题，并将这些错题信息整理成DataFrame格式以便进一步处理。

    参数:
    - subject (str): 需要回顾错题的科目。

    返回:
    - DataFrame: 包含所有错题信息的DataFrame，列包括'序号', '问题', '类型', '难度', '选项', '学科'。
    """
    # 日志记录，表示智能体开始回顾错题
    logger.info("智能体开始回顾错题... ...")

    # 初始化数据库助手对象，用于操作题目数据库
    db_helper = QuestionDBHelper()

    # 通过数据库助手从数据库中获取指定科目的所有错题
    questions = db_helper.get_questions_by_subject(subject)

    # 将查询结果转换为DataFrame，以便于查看和操作
    columns = ["序号", "问题", "类型", "难度", "选项", "学科"]
    data = pd.DataFrame(questions, columns=columns)

    # 日志记录，表示智能体结束回顾错题
    logger.info("智能体结束回顾错题... ...")

    # 返回整理后的错题DataFrame
    return pd.DataFrame(data)


def enhance_question(selected_index: gr.SelectData, dataframe_origin):
    """
    根据选定的表格行的问题，出同类的题目用于巩固。

    参数:
    - selected_index (gr.SelectData): 选定的表格行。

    返回:
    - str: 返回用于巩固的新题目。
    """

    # 使用日志记录错题巩固的开始
    logger.info("智能体开始错题巩固... ...")

    # 提取错题文本
    err_question_text = selected_index.row_value

    # 初始化增强模型，用于生成类似错题
    enhance_model = ChatOllama(
        base_url="http://localhost:11434",
        temperature=0.8,
        model=model_name
    )

    # 定义Prompt模板，指导模型基于错题生成新题目
    before_rag_template = "请参加这道题目（ {err_question_text} )的内容， 出1道同类的题目.如果你不知道答案，就回答不知道，不要试图编造答案。"
    before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)

    # 构建链式调用，将错题文本传递给模型以生成新题目
    after_rag_chain = (
            {"err_question_text": RunnablePassthrough()}
            | before_rag_prompt
            | enhance_model
            | StrOutputParser()
    )

    # 使用日志记录错题巩固的结束
    logger.info("智能体结束错题巩固... ...")

    # 调用链式结构，传入错题文本，返回生成的题目
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
            # 错题回顾
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
                    column_widths=[80, 300, 80, 80, 300, 80],
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
