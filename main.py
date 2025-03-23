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
import llm

import requests
import httpx
import feedparser
from bs4 import BeautifulSoup
from telegram import constants, InputMediaPhoto, Bot, error


llm_state = llm.new_state_gpt4free()
llm_state = llm.try_fetch_gpt4free(llm_state)
aclient = httpx.AsyncClient(
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/117.0"
    }
)
Post = Union[
    Dict[Literal["guid", "title", "link", "description"], str],
    Dict[
        Literal["guid", "title", "link", "description", "image"],
        str,
    ],
]
ScraperState = Dict[Literal["visited"], List[str]]

TARGET_FILEPATH = "ithome-filtered.rss"
CHECK_RESULT_FILEPATH = "check_result.json"
INIT_PROMPT = """\
# 身份

你是一个新闻过滤机器人，你需要帮助用户过滤一系列新闻。

# 流程

用户会给你一系列JSON格式的新闻的标题和摘要，你需要读取并分析其中的内容，并帮助用户判断是否为目标信息。

你的流程如下：

1. 分析新闻的内容，并使用一句话总结。
2. 根据下一部分的规则判断新闻是否为目标信息。
3. 判断其是不是目标信息，并输出为needed_news，类型为bool
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
    "needed_news": true,
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
        "content": "IT之家 3 月 23 日消息，中国发展高层论坛 2025 年年会今日在北京召开，中国新闻社记者询问宇树科技创始人王兴兴：“家用人形机器人何时上市？”王兴兴坦言：“其实我们目前像工业端会发展更快一点，家用还是会更慢一点，大家都在推进这个事情，但是具体多长时间，也不是特别好预估，我觉得，也不是最近两三年可以实现的问题。”宇树科技旗下的 Unitree H1 和 G1 人形机器人 2 月 12 日在京东线上首发开售，售价分别为\xa065 万元和 9.9 万元。不过在上架后不久便被下架。据IT之家此前报道，宇树\xa0Unitree H1 机器人于 2023 年 8 月首次公布，关键尺寸为（1520+285）/570/220mm，大腿和小腿长度 400mm×2，手臂总长度 338mm×2；关节单元极限扭矩：膝关节约 360N・m、髋关节约 220N・m、踝关节约 45N・m、手臂关节约 75N・m；行走速度大于 1.5m/s，潜在运动能力＞5m/s；内置 15Ah 电池，最大电压 67.2V。宇树 G1\xa0人形机器人于 2024 年 5 月发布，定价\xa09.9 万元起，官方描述为“人形智能体、AI 化身”。该机器人体重约 35kg、身高约 127cm，拥有 23~43 个关节电机，关节最大扭矩 120N・m；支持模仿 & 强化学习驱动，“在 AI 加速下的机器人技术，每天都在升级进化”。",
    },
    {
        "role": "assistant",
        "content": '{\n    "summerize": "这篇文章讨论了宇树科技创始人王兴兴对家用人形机器人上市时间的看法，表示在最近两三年内无法实现。",\n    "analyze": [\n        "不包含新款产品信息",\n        "不是产品开卖信息",\n        "不是电影上线信息",\n        "不是手机系统更新",\n        "不是汽车新闻",\n        "不是软件更新"\n    ],\n    "needed_news": true,\n    "analyze_tags": [\n        "AI: 和人形机器人及AI相关",\n        "机器人: 讨论了人形机器人",\n        "无人机: 和无人机无关",\n        "芯片: 和芯片无关",\n        "游戏: 和游戏无关",\n        "互联网: 和互联网无关",\n        "饮食: 和饮食无关",\n        "数码: 和数码无关",\n        "汽车: 和汽车无关",\n        "国内: 和国内有关，涉及中国",\n        "国外: 和国外无关",\n        "港台: 和港台无关"\n    ],\n    "extra_tags": [\n        "宇树科技"\n    ],\n    "tags": [\n        "AI", "机器人", "国内", \n        "宇树科技"\n    ]\n}',
    },
]

HTML_MESSAGE = """
<a href="{link}">{title}</a>

{description}
"""


class HTTPFailed(Exception):
    """HTTP请求错误"""


class ParseFeedFailed(Exception):
    """解析信息错误"""


class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = ""

    def handle_data(self, data):
        self.text += data


class Scraper:
    """RSS爬虫"""

    def __init__(self, filepath: str, state=None):
        self.filepath = filepath
        self.state: ScraperState = state if state is not None else {"visited": []}

    def filter_posts(self, posts: List[Post]) -> List[Post]:
        """过滤掉已经爬取的post

        Args:
            posts (List[Post]): 需要过滤的posts

        Returns:
            List[Post]: 过滤结果
        """
        return [post for post in posts if post["guid"] not in self.state["visited"]]

    def record_posts(self, posts: List[Post]):
        """将一系列post标记为已经爬取，去除太早期的记录

        Args:
            posts (List[Post]): 需要标记的post
        """
        self.state["visited"] += [post["guid"] for post in posts]
        self.state["visited"] = self.state["visited"][-500:]

    def fetch_new_posts(self) -> List[Post]:
        """获取新的posts

        Raises:
            HTTPFailed: HTTP失败
            ParseFeedFailed: 解析失败

        Returns:
            List[Post]: 爬取到的post
        """
        try:
            rss = feedparser.parse(self.filepath)
        except Exception as exc:
            raise ParseFeedFailed() from exc
        return [
            {
                "guid": item["id"],
                "title": item["title"],
                "link": item["link"],
                "description": item["summary"],
            }
            for item in rss.entries
        ]

    def find_image(self, post: Post) -> str | None:
        """找到一个post中的图片

        Args:
            post (Post): Post

        Returns:
            str | None: 图片链接
        """
        desc = post["description"]
        doc = BeautifulSoup(desc, "html.parser")
        img = doc.find("img")
        return img.attrs["src"] if img else None

    def new_posts(self) -> List[Post]:
        """获取当前的所有post，寻找其中的图片并更新当前状态

        Returns:
            List[Post]: 所有Post
        """
        posts = self.fetch_new_posts()
        logging.info("There's %d posts", len(posts))
        posts = self.filter_posts(posts)
        logging.info("There's %d new posts", len(posts))

        self.record_posts(posts)
        for post in posts:
            post["image"] = self.find_image(post)
        for data in posts:
            data["description"] = BeautifulSoup(data["description"], "html.parser").text
        logging.info("Done finding new posts")
        return posts

    def save(self, file):
        """保存状态到文件中

        Args:
            file (File): 打开的文件，不会主动关闭
        """
        json.dump(self.state, file, indent=2)

    def load(self, file):
        """从文件中加载状态

        Args:
            fp (_type_): 文件
        """
        self.state = json.load(file)

    def refuse(self, post: Post):
        """从状态中删除之前爬取过的Post

        Args:
            post (Post): 需要删除的Post
        """
        if post["guid"] in self.state["visited"]:
            self.state["visited"].remove(post["guid"])


def convert_bytes_to_jpg(bytes_image):
    img = Image.open(BytesIO(bytes_image)).convert("RGBA")
    output = BytesIO()
    background = Image.new("RGB", img.size, (255, 255, 255)).convert("RGBA")
    img = Image.alpha_composite(background, img).convert("RGB")
    img.save(output, format="JPEG")
    jpg_bytes = output.getvalue()
    output.close()
    return jpg_bytes


async def parse_lines(lines_iter: Iterable[bytes]) -> Generator[str, None, None]:
    """从openai的stream数据流中解析出所有字符

    Args:
        lines_iter (Iterable[bytes]): 所有行的迭代器

    Raises:
        UnknownError: 解析错误

    Yields:
        Generator[str, None, None]: 返回的每一个字符
    """
    async for line in lines_iter:
        if line == "":
            continue
        if not line.startswith("data: "):
            raise llm.UnknownError("wrong line: " + line)
        line = line.removeprefix("data: ")
        if line == "[DONE]":
            break
        data = None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError as exp:
            raise llm.UnknownError() from exp

        if "choices" not in data:
            raise llm.UnknownError(f"Wrong data: {data}")
        choice = data["choices"][0]
        if choice.get("finish_reason"):
            if choice.get("finish_reason") == "stop":
                return
            raise llm.UnknownError(
                "wrong finish reason: " + choice.get("finish_reason")
            )
            # return
        yield choice["delta"].get("content", "")


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


def get_html_text(html):
    parser = MyHTMLParser()
    parser.feed(html)

    text = parser.text.strip()
    return text


def check_result_needed(guid):
    if not os.path.exists(CHECK_RESULT_FILEPATH):
        return None
    with open(CHECK_RESULT_FILEPATH, "r") as f:
        data = json.load(f)
        if guid not in data:
            return None
        result, _ = data[guid]
        return result


def check_result_add(guid, result):
    print(f"{result=} {guid=}")
    data = {}
    if os.path.exists(CHECK_RESULT_FILEPATH):
        with open(CHECK_RESULT_FILEPATH, "r") as f:
            data = json.load(f)
    data = {
        k: (k_result, add_time)
        for k, (k_result, add_time) in data.items()
        if time.time() - add_time < 86400 * 5
    }
    data[guid] = (result, int(time.time()))
    with open(CHECK_RESULT_FILEPATH, "w") as f:
        json.dump(data, f)


async def is_target_entry(title, desc_text) -> tuple[bool | None, list]:
    try:
        prompt = json.dumps(
            {"title": title, "description": desc_text}, ensure_ascii=False
        )
        chatgpt_answer = await chatgpt_ask(prompt)
        assert chatgpt_answer is not None
        print(f"{desc_text=}, {chatgpt_answer=}")
        data = json.loads(chatgpt_answer)
        is_useful = data.get("needed_news", True)
        tags = list(set(data.get("tags", []) + data.get("extra_tags", [])))
        print(f"{is_useful=} {title=} {tags=}")
        return is_useful, tags
    except Exception:
        print_exc()
        return None, []


def force_find(element, xpath):
    result = element.find(xpath)
    assert result is not None
    return result


async def send_post(bot: Bot, chat_id: int | str, post: Post) -> bool:
    """发送post到telegram

    Args:
        bot (Bot): telegram bot实例
        chat_id (int | str): 发送目标
        post (Post): 需要发送的post

    Returns:
        bool: 是否发送成功
    """
    media = None
    if post.get("image", None):
        content = None
        for _ in range(3):
            try:
                resp = await aclient.get(post["image"], timeout=10)
                assert resp.status_code == 200
                content = resp.content
                content = convert_bytes_to_jpg(content)
            except Exception:
                await asyncio.sleep(1)
                continue
        if content is None:
            return False
        media = [
            InputMediaPhoto(
                content,
                caption=HTML_MESSAGE.format(
                    title=post["title"],
                    description=post["description"],
                    link=post["link"],
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
                        title=post["title"],
                        description=post["description"],
                        link=post["link"],
                    ),
                    parse_mode=constants.ParseMode.HTML,
                )
            ]
    for _ in range(10):
        try:
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
                post_without_img = post.copy()
                del post_without_img["image"]
                return await send_post(bot, chat_id, post_without_img)
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


async def get_filtered_rss():
    r = requests.get(
        "https://www.ithome.com/rss",
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/117.0"
        },
        timeout=60,
    )
    tree = ET.parse(StringIO(r.text))

    root = tree.getroot()
    chan = root.find("channel")
    assert chan is not None, f"{r.status_code=} {r.text}"

    async def handle_element(element):
        title = force_find(element, "title").text
        desc = force_find(element, "description").text
        guid = force_find(element, "guid").text
        cached_result = check_result_needed(guid)
        desc = get_html_text(desc)
        if cached_result is not None:
            if not cached_result:
                chan.remove(element)
        else:
            is_needed, tags = await is_target_entry(title, desc)
            check_result_add(guid, is_needed)
            if not is_needed:
                chan.remove(element)
            else:
                for child in element.findall("description"):
                    child.text = " ".join("#" + tag.replace(" ", "_") for tag in tags)

    tasks = []
    for element in chan.findall("item"):
        task = asyncio.create_task(handle_element(element))
        tasks.append(task)
        await asyncio.sleep(0.5)
    await asyncio.gather(*tasks)
    tree.write(TARGET_FILEPATH, encoding="utf-8", xml_declaration=False)


async def send_post_or_refuse(scraper, bot, chat_id, post):
    result = await send_post(bot, chat_id, post)
    if not result:
        scraper.refuse(post)
    with open("scraper.json", "w", encoding="utf-8") as file:
        scraper.save(file)


async def post_to_telegram(bot, chat_id):
    posts = None
    scraper = Scraper(TARGET_FILEPATH)
    with open("scraper.json", "r", encoding="utf-8") as file:
        scraper.load(file)
    try:
        posts = scraper.new_posts()
    except Exception:
        print_exc()
        return
    tasks = []
    for post in posts:
        logging.info("Sending post %s", post["guid"])
        task = asyncio.create_task(send_post_or_refuse(scraper, bot, chat_id, post))
        tasks.append(task)
        await asyncio.sleep(0.5)
    await asyncio.gather(*tasks)


async def main():
    bot = Bot(os.environ["TELEGRAM_BOT_TOKEN"])
    await get_filtered_rss()
    await post_to_telegram(bot, chat_id="@ithome_aifilter")


if __name__ == "__main__":
    asyncio.run(main())
