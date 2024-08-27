# 导入sqlite3模块，用于数据库操作
import sqlite3

class QuestionDBHelper:
    """
    数据库帮助器类，用于处理与考试问题相关的数据库操作。

    属性:
    db_name -- 数据库文件名
    """

    def __init__(self, db_name='dbs/exam_questions.db'):
        """
        初始化数据库帮助器类。

        参数:
        db_name -- 数据库文件名，默认为'exam_questions.db'
        """
        self.db_name = db_name

    def _connect(self):
        """
        建立到数据库的连接。

        返回:
        连接对象和游标对象的元组
        """
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        return conn, cursor

    def insert_question(self, text, question_type, difficulty, answers, subject):
        """
        将问题插入到数据库中。

        参数:
        text -- 问题的文本内容
        question_type -- 问题的类型（例如，选择题、填空题）
        difficulty -- 问题的难度级别
        answers -- 问题的正确答案
        subject -- 问题所属的科目或主题

        返回:
        无
        """
        # 连接到数据库
        conn, cursor = self._connect()
        # 假设我们有以下学科标签
        labels = ["语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "其它"]
        if subject is None:
            subject = labels[8]
        # 执行INSERT语句，将新的问题插入到'questions'表中
        cursor.execute('''
            INSERT INTO questions (text, type, difficulty, answers, subject) 
            VALUES ( ?, ?, ?, ?, ?)
        ''', (text, question_type, difficulty, answers, subject))

        # 提交当前事务，以保存更改
        conn.commit()
        # 关闭数据库连接，释放资源
        conn.close()

    def get_questions_by_subject(self, subject):
        """
        获取指定科目的所有问题。

        参数:
        subject -- 科目名称

        返回:
        包含问题记录的列表
        """
        # 连接到数据库
        conn, cursor = self._connect()

        # 查询指定科目的所有问题
        cursor.execute('SELECT * FROM questions WHERE subject = ?', (subject,))
        questions = cursor.fetchall()

        # 关闭数据库连接，释放资源
        conn.close()

        return questions

    def update_question(self, question_id, text=None, question_type=None, difficulty=None, answers=None, subject=None):
        """
        更新指定问题的信息。

        参数:
        question_id -- 要更新的问题的ID
        text -- 新的问题文本（可选）
        question_type -- 新的问题类型（可选）
        difficulty -- 新的难度级别（可选）
        answers -- 新的答案（可选）
        subject -- 新的科目（可选）

        返回:
        无
        """
        # 构建SET子句
        set_clause = []
        values = []

        if text is not None:
            set_clause.append("text = ?")
            values.append(text)

        if question_type is not None:
            set_clause.append("type = ?")
            values.append(question_type)

        if difficulty is not None:
            set_clause.append("difficulty = ?")
            values.append(difficulty)

        if answers is not None:
            set_clause.append("answers = ?")
            values.append(answers)

        if subject is not None:
            set_clause.append("subject = ?")
            values.append(subject)

        # 如果没有要更新的字段，则直接返回
        if not set_clause:
            return

        # 连接到数据库
        conn, cursor = self._connect()

        # 构建完整的UPDATE语句
        set_clause_str = ", ".join(set_clause)
        query = f"UPDATE questions SET {set_clause_str} WHERE id = ?"
        values.append(question_id)

        # 执行UPDATE语句
        cursor.execute(query, tuple(values))

        # 提交当前事务，以保存更改
        conn.commit()
        # 关闭数据库连接，释放资源
        conn.close()

    def delete_question(self, question_id):
        """
        删除指定的问题。

        参数:
        question_id -- 要删除的问题的ID

        返回:
        无
        """
        # 连接到数据库
        conn, cursor = self._connect()

        # 执行DELETE语句
        cursor.execute('DELETE FROM questions WHERE id = ?', (question_id,))

        # 提交当前事务，以保存更改
        conn.commit()
        # 关闭数据库连接，释放资源
        conn.close()
