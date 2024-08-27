一、项目简介
1、背景与目标
    错题整理是学习中起重要作用，但是因错题整理较为费时，许多学生在错题整理方面并不重视。因此，帮助学生们更加高效地整理错题成为了一个重要的课题。随着生成式人工智能的迅速发展，大语言模型展现出出色的文字识别和理解语意的能力。
 本项目研究目的是利用大语言模型开发一个可以帮助学生们更加高效地整理错题的AI助手，针对学生的个性化情况进行教学。这个研究中，我们重点针对中学生学习中的试题讲解，错题纠正，薄弱环节巩固等需求进行开发。满足能够对题目进行解答与讲解，以及进一步给出新题。并能将历史学习数据进行整理记录，形成个人的数据库、错题集。
2、主要研究内容
    错题整理助手需要基于OCR识别技术将图片中的题目转换成文本、利用大语言模型对题目进行解析并详细讲解，能够基于自然语言理解对错题按学科归类，并，还可以利用AIGC生成出新的题目。同时，以上能力需要通过大模型应用开发框架实现错题录入界面交互、错题数据存储、大语言模型调用及整体功能组件的链式调用。

3、项目创新点
    在这个项目中，我们利用语言大模型开发了一个错题整理助手，旨在个性化的帮助学生高效整理错题。错题整理助手可以像老师一样根据学生的问题，提供个性化的答案和详解，并可以生成知识点相近的新题目，定期为学生提供练习，帮助学生进行巩固，实验表明错题整理助手答复的正确率较高，讲解也很详细有针对性。
    项目的创新点包括几个方面：
创新点一：错题整理助手通过设计并优化提示，以 LangChain 作为开发框架，将大语言模型的语言能力转化为一个可以解题、讲题并出题的错题整理助手，并在实践中取得了优异的效果。
创新点二：错题整理助手可以针对学生的错题和提问进行回答，并可以出新题，定期为学生提供练习，帮助学生巩固知识，相比之前依赖固定题库和讲解的工具，错题整理助手个性化讲解有利于提升学生的学习效率。
创新点三：错题整理助手采用多种技术提升效率和用户体验，包括使用OCR识别技术来解决手动输入题目过程繁琐、效率低下的问题，使用Gradio 开发学生友好的用户界面。
创新点四：错题整理助手采用~~~对错题进行归类，使学生的错题集分学科，有条理，便于学生练习和复习。



二、项目实现流程
    错题整理助手利用OCR识别技术将图片中的题目转换成文本，利用大语言模型对错题目进行分析讲解，错题分类整理，并举一反三出新的题目。
主要步骤：（1）题目识别：将拍摄错题图片识别为题目文本，解决手动输入题目过程繁琐、效率低下的问题。（2）题目讲解：根据用户提供的需求（自定义的讲解需求），调用大语言模型，讲解该题目的正确解题思路，让错题整理助手可以像老师一样根据学生的问题，提供个性化的答案和详解。（3）题目储存：根据题目所属的学科进行智能归类，按学科储存，方便学生对错题进行回顾。（4）出新的题目：每隔一天出一道类似、知识点相近的题目，定期为学生提供练习，帮助学生进行巩固。


三、项目功能实现
3.1 错题识别
设计意图: 将拍摄错题图片识别为题目文本，解决手动输入题目过程繁琐、效率低下的问题。
实现方法: 使用Google的pytesseract从错题图像中提取错题文本。
功能演示: 如果可能，提供功能演示的描述或链接，以便评审团理解功能的实际表现。
3.2 错题解析
设计意图: 对题目进行解析，使学生学会做与错题同类的题型
实现方法: 利用大语言模型生成解析
功能演示: 提供功能演示的描述或链接。
3.3 错题分类
设计意图: 对错题按学科归类，使学生查找错题时更加方便
实现方法: 利用大语言模型，基于自然语言理解对错题按学科归类
功能演示: 提供功能演示的描述或链接。
3.4 错题存储
设计意图: 根据项目实际情况，继续详细描述其他功能的设计意图、实现方法和演示
实现方法: 
功能演示: 提供功能演示的描述或链接。
3.5 错题巩固
设计意图: 举一反三，生成出新的题目，让学生进一步了解同类题型
实现方法: 利用AIGC生成出新的题目
功能演示: 提供功能演示的描述或链接。