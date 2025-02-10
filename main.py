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
2. 根据第三部分的规则判断新闻是否为目标信息。
3. 检查上方的分析和判断是否正确，如果不正确请分析原因。
4. 判断其是不是目标信息，并输出为needed_news，类型为bool

# 规则

- 标题包含新款产品信息和其价格的不是目标信息。
- 某某产品开卖不是目标信息。
- 电影上线信息不是目标信息。
- 手机系统更新不是目标信息。
- 和汽车相关的新闻，尤其是宝马、法拉第未来、比亚迪、极狐汽车等汽车品牌相关的信息不是目标信息。
- 软件更新新版本不是目标信息。
- 商店打折，京东、淘宝、拼多多优惠活动不是目标信息。
- 不匹配以上规则的所有新闻都是目标信息

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
    "check": [
        "确实不包含新款产品信息",
        "确实不是产品开卖信息",
        "确实不是电影上线信息",
        "确实不是手机系统更新",
        "确实不是汽车新闻",
        "确实不是软件更新",
    ],
    "final_answer": "是目标信息",
    "needed_news": true
}
"""

LLM_EXTRA_MESSAGES: llm.Messages = [
    {
        "role": "user",
        "content": """
{"title": "问界新 M7 车型今日大定超 2600 台，消息称本月品牌新增订单突破 3 万", "description": "IT之家 9 月 30 日消息，这个月内，问界新 M7 的热度持续维持着有增无减的态势。自从该车上市后，日均订单超过 1500 台，大定累计已经超过 2 万台。目前最新的数据是 9 月 29 日、30 日的，其中 29 日大定超过 2400 台，30 日的大定数量则突破了 2600 台。这就相当于昨天今天两日之内，问界新 M7 的大定数量超过 5000 台。IT之家此前报道，该车曾在 9 月 17 日取得本月大定数量记录：2700 台。目前来看，问界似乎已经形成了“每逢破 2000 必发海报庆祝”的传统。汽车销售情况分析媒体“车 fans”创始人 @孙少军 09 分析称，截至今晚，问界品牌全系车型新增订单量已突破了 3 万。同时他表示，华为 Mate 系列机型也给问界带来了“排山倒海”的热度，新增订单量环比上月暴涨 8 倍。当然了，明天作为 10 月的第 1 天，不出意外的话也是各路车企集中汇报 9 月销量的日子。问界品牌 9 月的销量究竟如何，最快将在明天之内见分晓，IT之家将持续关注。问界新 M7 搭载华为 ADS 2.0 高阶智能驾驶系统，可实现不依赖高精地图的高速、城区智能驾驶，并号称预计今年 12 月城区 NCA 实现“全国都能开”。新车配备 1 个顶置激光雷达、3 个毫米波雷达、11 个高清视觉感知摄像头及 12 个超声波雷达等 27 个感知硬件，支持园区代客泊车、超窄车位泊车。"}
""".strip(),
    },
    {
        "role": "assistant",
        "content": """
{"summerize": "这是一篇关于问界新M7车型销售情况的文章，今年9月份订单数量突破了3万台。其中，最新数据显示，9月29日和30日的大定数量分别超过2400台和2600台。同时，华为Mate系列机型的推出也给问界品牌带来了热度，新增订单量环比上月暴涨8倍。明天将是各大车企集中发布9月销量的日子，IT之家将持续关注问界品牌的销售情况。问界新M7车型搭载了华为ADS 2.0高阶智能驾驶系统，拥有多项高科技感知硬件。", "analyze": ["不包含新款产品信息", "不是产品开卖信息", "不是电影上线信息", "不是手机系统更新", "这片文章是有关新品汽车上市的新闻，属于非目标信息中的汽车新闻", "不是软件更新"], "check": ["确实不包含新款产品信息", "确实不是产品开卖信息", "确实不是电影上线信息", "确实不是手机系统更新", "问界是一个汽车品牌，属于非目标新闻中的汽车新闻", "确实不是软件更新"], "final_answer": "是无用信息", "needed_news": false}
""".strip(),
    },
]

HTML_MESSAGE = """
<a href="{link}">{title}</a>
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


async def is_target_entry(title, desc_text):
    try:
        prompt = json.dumps(
            {"title": title, "description": desc_text}, ensure_ascii=False
        )
        data = await chatgpt_ask(prompt)
        assert data is not None
        data = json.loads(data)
        is_useful = data.get("needed_news", True)
        print(f"{is_useful=} {title=}")
        return is_useful
    except Exception:
        print_exc()
        return True


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
                caption=HTML_MESSAGE.format(title=post["title"], link=post["link"]),
                parse_mode=constants.ParseMode.HTML,
            )
        ]
    else:
        with open("RSSFilter.png", "rb") as file:
            media = [
                InputMediaPhoto(
                    file.read(),
                    caption=HTML_MESSAGE.format(title=post["title"], link=post["link"]),
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
            result = await is_target_entry(title, desc)
            check_result_add(guid, result)
            if not result:
                chan.remove(element)

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
