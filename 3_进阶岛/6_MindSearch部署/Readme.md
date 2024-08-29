MindSearch CPU-only 版部署,不是指将LLM进行CPU部署

是采用LLM服务+web模式联合部署，web前端即可以使用CPU部署

目前LLM有免费公共的服务接口，CPU-server也有免费资源，二者结合就可以实现白嫖！

# 1、白嫖LLM服务

此处将白嫖 硅基流动的API Key

打开 https://account.siliconflow.cn/login 注册账号

打开 https://cloud.siliconflow.cn/account/ak 创建新 API 密钥，复制保存

OK 到此结束

# 2、白嫖CPU资源

首先打开 https://huggingface.co/spaces 并点击 Create new Space选择Gradio 和FREE的空间来部署服务。

![img.png](img.png)

OK 到此结束

# 3、白嫖组合

白嫖组合的思路就是 利用白嫖的CPU资源部署web来调用白嫖的LLM API

但是LLM的API密钥直接写入代码是非常危险的

所以huggingface为我们考虑好了，进入空间配置界面，点击“Settings”

或者如下链接：https://huggingface.co/spaces/<你的名字>/<仓库名称>/settings

在“Variables and secrets”中创建新第三方密钥

name 一栏输入 SILICON_API_KEY，value 一栏输入 硅基流动的API Key内容

![img_1.png](img_1.png)

以上就完成了前端安全访问API_KEY的安全配置。

<i><b>以下是方便在本地网络环境中实现huggingface上传文件等操作,本地环境为windows,部分命令不适用Linux
</i></b>

下载源码文件

```commandline
git clone https://github.com/InternLM/MindSearch.git
```

新建一个目录 mindsearch_deploy

将 MindSearch/mindsearch 复制到 mindsearch_deploy中

将 MindSearch/requirements.txt 复制到 mindsearch_deploy中

在mindsearch_deploy中新建一个app.py文件
复制内容如下：

```commandline
import json
import os

import gradio as gr
import requests
from lagent.schema import AgentStatusCode

os.system("python -m mindsearch.app --lang cn --model_format internlm_silicon &")

PLANNER_HISTORY = []
SEARCHER_HISTORY = []


def rst_mem(history_planner: list, history_searcher: list):
    '''
    Reset the chatbot memory.
    '''
    history_planner = []
    history_searcher = []
    if PLANNER_HISTORY:
        PLANNER_HISTORY.clear()
    return history_planner, history_searcher


def format_response(gr_history, agent_return):
    if agent_return['state'] in [
            AgentStatusCode.STREAM_ING, AgentStatusCode.ANSWER_ING
    ]:
        gr_history[-1][1] = agent_return['response']
    elif agent_return['state'] == AgentStatusCode.PLUGIN_START:
        thought = gr_history[-1][1].split('```')[0]
        if agent_return['response'].startswith('```'):
            gr_history[-1][1] = thought + '\n' + agent_return['response']
    elif agent_return['state'] == AgentStatusCode.PLUGIN_END:
        thought = gr_history[-1][1].split('```')[0]
        if isinstance(agent_return['response'], dict):
            gr_history[-1][
                1] = thought + '\n' + f'```json\n{json.dumps(agent_return["response"], ensure_ascii=False, indent=4)}\n```'  # noqa: E501
    elif agent_return['state'] == AgentStatusCode.PLUGIN_RETURN:
        assert agent_return['inner_steps'][-1]['role'] == 'environment'
        item = agent_return['inner_steps'][-1]
        gr_history.append([
            None,
            f"```json\n{json.dumps(item['content'], ensure_ascii=False, indent=4)}\n```"
        ])
        gr_history.append([None, ''])
    return


def predict(history_planner, history_searcher):

    def streaming(raw_response):
        for chunk in raw_response.iter_lines(chunk_size=8192,
                                             decode_unicode=False,
                                             delimiter=b'\n'):
            if chunk:
                decoded = chunk.decode('utf-8')
                if decoded == '\r':
                    continue
                if decoded[:6] == 'data: ':
                    decoded = decoded[6:]
                elif decoded.startswith(': ping - '):
                    continue
                response = json.loads(decoded)
                yield (response['response'], response['current_node'])

    global PLANNER_HISTORY
    PLANNER_HISTORY.append(dict(role='user', content=history_planner[-1][0]))
    new_search_turn = True

    url = 'http://localhost:8002/solve'
    headers = {'Content-Type': 'application/json'}
    data = {'inputs': PLANNER_HISTORY}
    raw_response = requests.post(url,
                                 headers=headers,
                                 data=json.dumps(data),
                                 timeout=20,
                                 stream=True)

    for resp in streaming(raw_response):
        agent_return, node_name = resp
        if node_name:
            if node_name in ['root', 'response']:
                continue
            agent_return = agent_return['nodes'][node_name]['detail']
            if new_search_turn:
                history_searcher.append([agent_return['content'], ''])
                new_search_turn = False
            format_response(history_searcher, agent_return)
            if agent_return['state'] == AgentStatusCode.END:
                new_search_turn = True
            yield history_planner, history_searcher
        else:
            new_search_turn = True
            format_response(history_planner, agent_return)
            if agent_return['state'] == AgentStatusCode.END:
                PLANNER_HISTORY = agent_return['inner_steps']
            yield history_planner, history_searcher
    return history_planner, history_searcher


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">MindSearch Gradio Demo BY Huifeiyang</h1>""")   # 修改名字美化界面
    gr.HTML("""<p style="text-align: center; font-family: Arial, sans-serif;">MindSearch is an open-source AI Search Engine Framework with Perplexity.ai Pro performance. You can deploy your own Perplexity.ai-style search engine using either closed-source LLMs (GPT, Claude) or open-source LLMs (InternLM2.5-7b-chat).</p>""")
    gr.HTML("""
    <div style="text-align: center; font-size: 16px;">
        <a href="https://github.com/InternLM/MindSearch" style="margin-right: 15px; text-decoration: none; color: #4A90E2;">🔗 GitHub</a>
        <a href="https://arxiv.org/abs/2407.20183" style="margin-right: 15px; text-decoration: none; color: #4A90E2;">📄 Arxiv</a>
        <a href="https://huggingface.co/papers/2407.20183" style="margin-right: 15px; text-decoration: none; color: #4A90E2;">📚 Hugging Face Papers</a>
        <a href="https://huggingface.co/spaces/internlm/MindSearch" style="text-decoration: none; color: #4A90E2;">🤗 Hugging Face Demo</a>
    </div>
    """)
    with gr.Row():
        with gr.Column(scale=10):
            with gr.Row():
                with gr.Column():
                    planner = gr.Chatbot(label='planner',
                                         height=700,
                                         show_label=True,
                                         show_copy_button=True,
                                         bubble_full_width=False,
                                         render_markdown=True)
                with gr.Column():
                    searcher = gr.Chatbot(label='searcher',
                                          height=700,
                                          show_label=True,
                                          show_copy_button=True,
                                          bubble_full_width=False,
                                          render_markdown=True)
            with gr.Row():
                user_input = gr.Textbox(show_label=False,
                                        placeholder='帮我搜索一下 InternLM 开源体系',
                                        lines=5,
                                        container=False)
            with gr.Row():
                with gr.Column(scale=2):
                    submitBtn = gr.Button('Submit')
                with gr.Column(scale=1, min_width=20):
                    emptyBtn = gr.Button('Clear History')

    def user(query, history):
        return '', history + [[query, '']]

    submitBtn.click(user, [user_input, planner], [user_input, planner],
                    queue=False).then(predict, [planner, searcher],
                                      [planner, searcher])
    emptyBtn.click(rst_mem, [planner, searcher], [planner, searcher],
                   queue=False)

demo.queue()
demo.launch(server_name='0.0.0.0',
            server_port=7860,
            inbrowser=True,
            share=True)
```

在 huggingface 个人界面内 创建一个可写密钥 用于上传，将密钥保存，后面会用

![img_2.png](img_2.png)

本地git 一下空间的库信息

```commandline
git clone https://huggingface.co/spaces/<你的名字>/<仓库名称>
```

然后将 mindsearch_deploy 文件 全部复制到仓库中，参考如下：

（红框是我的space名称，仅作参考）

![img_3.png](img_3.png)

cmd cd 到空间内

```commandline
git add .
git commit -m "update"
git push
```

最后一步 push 会弹出验证框

其中用户名为<你的名字>

密码为<可写密钥>

然后等待推送完成，如下图

![img_4.png](img_4.png)

然后点击自己的链接：https://huggingface.co/spaces/TaylorYang/MindSearch

可以愉快的享用白嫖的喜悦了，顺便问问白嫖的快乐秘密

![img_5.png](img_5.png)

