VScode DEBUG纪录：

###对关键变量位置做标记###

关键变量：

去标点字符串；小写字符串；分词列表

<img src="https://github.com/huifeiyang/InternLM_Tutorial_3/blob/master/1_%E5%85%A5%E9%97%A8/2_Python/DEBUG%E8%AE%B0%E5%BD%95/1_%E8%AE%BE%E7%BD%AE%E6%96%AD%E7%82%B9.png?raw=true"/>

###跟踪流程###

其中对动态变量做观察

<img src="https://github.com/huifeiyang/InternLM_Tutorial_3/blob/master/1_%E5%85%A5%E9%97%A8/2_Python/DEBUG%E8%AE%B0%E5%BD%95/2_%E8%B7%9F%E8%B8%AA%E6%B5%81%E7%A8%8B.png?raw=true"/>


###发现问题###

对英语缩写没有区分

<img src="https://github.com/huifeiyang/InternLM_Tutorial_3/blob/master/1_%E5%85%A5%E9%97%A8/2_Python/DEBUG%E8%AE%B0%E5%BD%95/3_%E5%8F%91%E7%8E%B0%E9%97%AE%E9%A2%98.png?raw=true"/>


###编写前置处理###

将 's 换为 is

<img src="https://github.com/huifeiyang/InternLM_Tutorial_3/blob/master/1_%E5%85%A5%E9%97%A8/2_Python/DEBUG%E8%AE%B0%E5%BD%95/4_%E4%BF%AE%E6%AD%A3%E4%BB%A3%E7%A0%81.png?raw=true"/>



###重运行检查###

<img src="https://github.com/huifeiyang/InternLM_Tutorial_3/blob/master/1_%E5%85%A5%E9%97%A8/2_Python/DEBUG%E8%AE%B0%E5%BD%95/5_%E8%BE%93%E5%87%BA%E5%B1%95%E7%A4%BA.png?raw=true"/>