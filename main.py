import asyncio
import os
import json
import logging
import time
from io import StringIO, BytesIO
from xml.etree import ElementTree as ET
from html.parser import HTMLParser
from typing import List, Dict, Literal, Union, Generator, Tuple, Iterable
from traceback import print_exc
from copy import deepcopy
from PIL import Image
from pathlib import Path
import llm

import httpx
import feedparser
from bs4 import BeautifulSoup
from telegram import constants, InputMediaPhoto, Bot, error

telegram_lock = asyncio.Lock()
llm_state = llm.new_state_gpt4free()
aclient = httpx.AsyncClient(
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/117.0"
    }
)

HTML_MESSAGE = """
<a href="{link}">{title}</a>

{description}
"""

INIT_PROMPT = """\
# 身份

你是一个新闻过滤机器人，你需要帮助用户过滤一系列新闻。

# 流程

用户会给你一系列JSON格式的新闻的标题和摘要，你需要读取并分析其中的内容，并帮助用户判断是否为目标信息。

你的流程如下：

1. 分析新闻的内容，并使用一句话总结。
2. 根据下一部分的规则判断新闻是否为目标信息。
3. 判断其是不是目标信息，并输出为is_useful_news，类型为bool
4. 根据下面的所有标签，用一句话判断每个标签是否和新闻相符合
5. 列出提到的所有公司、国家和地区的中文名（如苹果、美国、香港等），作为额外的标签（extra_tags）输出
6. 根据下面的所有标签给各个新闻打上标签，并加上extra_tags的内容，输出为JSON列表

# 规则

- 标题包含新款产品信息和其价格的不是目标信息。
- 某某产品开卖不是目标信息。
- 电影上线信息不是目标信息。
- 手机系统更新不是目标信息。
- 软件更新新版本不是目标信息。
- 商店打折，京东、淘宝、拼多多优惠活动不是目标信息。
- 不匹配以上规则的所有新闻都是目标信息

# 所有标签

- AI: 和大语言模型、AIGC、芯片等领域相关，和CUDA等技术相关，以及英伟达等GPU公司、OpenAI, DeepSeek等AI技术公司相关
- 机器人: 和人形机器人等机器人相关的新闻
- 无人机: 和无人机相关，或者和大疆等公司的新闻
- 芯片: 和芯片相关的新闻
- 游戏: 和电子游戏相关的，和索尼、任天堂等游戏公司相关的新闻，特别注意英伟达与AMD的新闻不属于此类
- 互联网: 和互联网行业，或者和腾讯、阿里巴巴等互联网公司相关的新闻
- 饮食: 和可口可乐、瑞幸咖啡等饮料相关，或者和食品相关的新闻
- 数码: 和手机、电脑、相机、游戏机等数码产品相关的新闻
- 汽车: 和宝马、特斯拉、比亚迪、小米汽车等汽车相关的新闻
- 网安: 软件漏洞，黑客攻击等    和网络安全相关的新闻
- 国内: 和中华人民共和国有关的新闻
- 国外: 和美国、欧洲国家、日本等外国相关的新闻
- 港台: 和香港、台湾相关的新闻

# 输入输出格式

用户的输入和你的输出都为JSON格式

示例：

用户：{
    "title": "因涉嫌 AI 芯片“反竞争”行为，消息称欧盟对英伟达展开早期调查", 
    "description": "IT之家 9 月 30 日消息，据彭博社援引“熟悉内情”的人士称，欧盟正在对 AI 芯片市场涉嫌“反竞争”的滥用行为展开早期调查，其中英伟达占据主导地位。据报道，欧盟委员会一直以非正式手段来收集有关“GPU 领域潜在滥用市场优势地位行为”的意见，以了解未来是否有必要进行干预。不过在初期阶段，这些调查可能并不会导致正式调查或处罚。消息人士透露称，法国当局也展开了类似的调查，并就英伟达在 AI 芯片领..."
}

机器人：{
    "summerize": "这是一篇有关欧盟调查英伟达“反竞争”行为的文章。据报道欧盟正收集有关“GPU领域潜在滥用市场优势地位行为”的意见。",
    "analyze": [
        "不包含新款产品信息",
        "不是产品开卖信息",
        "不是电影上线信息",
        "不是手机系统更新",
        "不是汽车新闻",
        "不是软件更新"
    ],
    "is_useful_news": true,
    "analyze_tags": [
        "AI: 和英伟达与AI芯片相关",
        "机器人: 和机器人无关",
        "无人机: 和无人机无关",
        "芯片: 提到芯片，和芯片有关",
        "游戏: 虽然与英伟达有关，但和电子游戏无关",
        "互联网: 和互联网无关",
        "饮食: 和饮食无关",
        "数码: 和数码无关",
        "汽车: 和汽车无关",
        "网安: 和网安无关",
        "国内: 和国内无关",
        "国外: 和欧盟有关，是与欧盟有关的新闻",
        "港台: 和港台无关"
    ],
    "extra_tags": [
        "苹果",
        "欧盟"
    ],
    "tags": [
        "AI", "芯片", "国外", 
        "苹果", "欧盟"
    ]
}
"""

LLM_EXTRA_MESSAGES: llm.Messages = [
    {
        "role": "user",
        "content": json.dumps(
            {
                "title": "中国发展高层论坛 2025 年年会今日在北京召开，宇树科技创始人答记者问",
                "description": "IT之家 3 月 23 日消息，中国发展高层论坛 2025 年年会今日在北京召开，中国新闻社记者询问宇树科技创始人王兴兴：“家用人形机器人何时上市？”王兴兴坦言：“其实我们目前像工业端会发展更快一点，家用还是会更慢一点，大家都在推进这个事情，但是具体多长时间，也不是特别好预估，我觉得，也不是最近两三年可以实现的问题。”宇树科技旗下的 Unitree H1 和 G1 人形机器人 2 月 12 日在京东线上首发开售，售价分别为\xa065 万元和 9.9 万元。不过在上架后不久便被下架。据IT之家此前报道，宇树\xa0Unitree H1 机器人于 2023 年 8 月首次公布，关键尺寸为（1520+285）/570/220mm，大腿和小腿长度 400mm×2，手臂总长度 338mm×2；关节单元极限扭矩：膝关节约 360N・m、髋关节约 220N・m、踝关节约 45N・m、手臂关节约 75N・m；行走速度大于 1.5m/s，潜在运动能力＞5m/s；内置 15Ah 电池，最大电压 67.2V。宇树 G1\xa0人形机器人于 2024 年 5 月发布，定价\xa09.9 万元起，官方描述为“人形智能体、AI 化身”。该机器人体重约 35kg、身高约 127cm，拥有 23~43 个关节电机，关节最大扭矩 120N・m；支持模仿 & 强化学习驱动，“在 AI 加速下的机器人技术，每天都在升级进化”。",
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "assistant",
        "content": '{\n    "summerize": "这篇文章讨论了宇树科技创始人王兴兴对家用人形机器人上市时间的看法，表示在最近两三年内无法实现。",\n    "analyze": [\n        "不包含新款产品信息",\n        "不是产品开卖信息",\n        "不是电影上线信息",\n        "不是手机系统更新",\n        "不是汽车新闻",\n        "不是软件更新"\n    ],\n    "is_useful_news": true,\n    "analyze_tags": [\n        "AI: 和人形机器人及AI相关",\n        "机器人: 讨论了人形机器人",\n        "无人机: 和无人机无关",\n        "芯片: 和芯片无关",\n        "游戏: 和游戏无关",\n        "互联网: 和互联网无关",\n        "饮食: 和饮食无关",\n        "数码: 和数码无关",\n        "汽车: 和汽车无关",\n        "国内: 和国内有关，涉及中国",\n        "国外: 和国外无关",\n        "港台: 和港台无关"\n    ],\n    "extra_tags": [\n        "宇树科技"\n    ],\n    "tags": [\n        "AI", "机器人", "国内", \n        "宇树科技"\n    ]\n}',
    },
]


def force_find_text(element, xpath):
    elem = element.find(xpath)
    assert elem is not None
    result = elem.text
    assert isinstance(result, str)
    return result


async def get_filtered_rss():
    r = httpx.get(
        "https://www.ithome.com/rss",
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/117.0"
        },
        timeout=10,
    )
    tree = ET.parse(StringIO(r.text))

    root = tree.getroot()
    chan = root.find("channel")
    assert chan is not None, f"{r.status_code=} {r.text}"
    return [
        {
            "title": force_find_text(element, "title"),
            "description": force_find_text(element, "description"),
            "link": force_find_text(element, "link"),
            "guid": force_find_text(element, "guid"),
        }
        for element in chan.findall("item")
    ]


async def chatgpt_ask(question):
    msg = llm.new_msg(INIT_PROMPT)
    msg += LLM_EXTRA_MESSAGES
    msg = llm.append(msg, "user", question)
    cfg = llm.new_config()
    for _ in range(10):
        try:
            answer = ""
            for delta, _, _ in llm.answer_stream_gpt4free(msg, cfg, llm_state):
                answer += delta
            return answer
        except llm.TooManyRequests:
            await asyncio.sleep(3)
    return None


async def chatgpt_transform(title, description):
    try:
        prompt = json.dumps(
            {"title": title, "description": description}, ensure_ascii=False
        )
        chatgpt_answer = await chatgpt_ask(prompt)
        if chatgpt_answer is None:
            return None
        data = json.loads(chatgpt_answer)
        is_useful_news = data.get("is_useful_news", True)
        tags = list(set(data.get("tags", []) + data.get("extra_tags", [])))
        return {
            "is_useful_news": is_useful_news,
            "tags": tags,
        }
    except Exception:
        return None


def find_image(html: str) -> str | None:
    doc = BeautifulSoup(html, "html.parser")
    Path("test.html").write_text(html)
    img = doc.find("img")
    return img.attrs["src"] if img else None


def convert_bytes_to_jpg(bytes_image):
    img = Image.open(BytesIO(bytes_image)).convert("RGBA")
    output = BytesIO()
    background = Image.new("RGB", img.size, (255, 255, 255)).convert("RGBA")
    img = Image.alpha_composite(background, img).convert("RGB")
    img.save(output, format="JPEG")
    jpg_bytes = output.getvalue()
    output.close()
    return jpg_bytes


async def send_post(
    bot: Bot,
    chat_id: int | str,
    title: str,
    description: str,
    link: str,
    image: str | None,
) -> bool:
    """发送post到telegram

    Args:
        bot (Bot): telegram bot实例
        chat_id (int | str): 发送目标
        post (Post): 需要发送的post

    Returns:
        bool: 是否发送成功
    """
    media = None
    if image is not None:
        content = None
        for _ in range(3):
            try:
                resp = await aclient.get(image, timeout=10)
                assert resp.status_code == 200
                content = resp.content
                content = convert_bytes_to_jpg(content)
            except Exception:
                await asyncio.sleep(5)
                continue
        if content is None:
            return False
        media = [
            InputMediaPhoto(
                content,
                caption=HTML_MESSAGE.format(
                    title=title,
                    description=description,
                    link=link,
                ),
                parse_mode=constants.ParseMode.HTML,
            )
        ]
    else:
        with open("RSSFilter.png", "rb") as file:
            media = [
                InputMediaPhoto(
                    file.read(),
                    caption=HTML_MESSAGE.format(
                        title=title,
                        description=description,
                        link=link,
                    ),
                    parse_mode=constants.ParseMode.HTML,
                )
            ]
    for _ in range(10):
        try:
            async with telegram_lock:
                await bot.send_media_group(
                    chat_id=chat_id,
                    media=media,
                )  # type: ignore
        except error.RetryAfter:
            print("rate limit")
            await asyncio.sleep(20)
            continue
        except error.TimedOut:
            return False
        except error.BadRequest as exp:
            if "image_process_failed" in str(exp):
                return await send_post(bot, chat_id, title, description, link, None)
            else:
                raise exp
        except error.TelegramError:
            print_exc()
            return False
        except Exception:
            print_exc()
            return False
        return True
    return False


async def main():
    bot = Bot(os.environ["TELEGRAM_BOT_TOKEN"])
    visited = []
    if Path("./visited.json").exists():
        visited = list(json.loads(Path("./visited.json").read_text()))
        visited = visited[-500:]

    new_items = await get_filtered_rss()
    new_items = [item for item in new_items if item["guid"] not in visited]
    print(f"{len(new_items)=}")
    chatgpt_answers = await asyncio.gather(
        *[chatgpt_transform(item["title"], item["description"]) for item in new_items]
    )
    print(f"{len(chatgpt_answers)=}")
    images = [
        find_image(item["description"] if isinstance(item["description"], str) else "")
        for item in new_items
    ]
    print(f"{len(images)=}")
    new_items_useful = [
        (
            item,
            image,
            " ".join("#" + tag.replace(" ", "_") for tag in chatgpt_transform["tags"]),
        )
        for item, chatgpt_transform, image in zip(new_items, chatgpt_answers, images)
        if chatgpt_transform
        if chatgpt_transform["is_useful_news"]
    ]
    print(f"{len(new_items_useful)=}")

    result = await asyncio.gather(
        *[
            send_post(
                bot,
                "@ithome_aifilter",
                news["title"],
                tags_str,
                news["link"],
                image,
            )
            for news, image, tags_str in new_items_useful
        ]
    )
    visited += [
        item["guid"]
        for item, chatgpt_answer in zip(new_items, chatgpt_answers)
        if chatgpt_answer is not None and not chatgpt_answer["is_useful_news"]
    ]
    visited += [
        news["guid"]
        for (news, _, _), is_success in zip(new_items_useful, result)
        if is_success
    ]
    for (news, _, _), is_success, chatgpt_answer in zip(
        new_items_useful, result, chatgpt_answers
    ):
        tags = chatgpt_answer["tags"] if chatgpt_answer is not None else None
        print(f"{news['title']=} {tags=} {is_success=}")

    Path("./visited.json").write_text(json.dumps(list(visited)))


if __name__ == "__main__":
    asyncio.run(main())
