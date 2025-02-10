"""所有LLM
"""

import json
from typing import (
    Dict,
    Literal,
    List,
    TypedDict,
    Tuple,
    Generator,
    Iterable,
    Callable,
    TypeVar,
)
import random
from copy import deepcopy
import logging
import os
import requests

# from g4f.client import Client # import when needed


logger = logging.getLogger("llm")

Message = Dict[Literal["content", "role"], str]
Messages = List[Message]


class Config(TypedDict):
    """给所有LLM的设置"""

    model: str
    temperature: float


Gpt4FreeState = dict


class EvilAPIState(TypedDict):
    """给EvilAPI的上下文"""

    urls: List[str]
    counts: Dict[str, int]


class EvilNextWebState(TypedDict):
    """给EvilNextWeb的上下文"""

    urls: List[str]
    counts: Dict[str, int]


State = EvilAPIState | EvilNextWebState | Gpt4FreeState
NewStateFunction = Callable[[], State]
AnswerFunction = Callable[[Messages, Config, State], Tuple[str, Messages, State]]
AnswerStreamFunction = Callable[
    [Messages, Config, State], Iterable[Tuple[str, Messages, State]]
]


class NormalContext(TypedDict):
    """正常上下文，提供进行一个对话需要的所有东西"""

    msg: Message
    cfg: Config
    state: State
    new_state: NewStateFunction
    answer: AnswerFunction


class LLMException(Exception):
    """这里所有错误的父类"""


class HTTPError(LLMException):
    """产生HTTP请求错误"""


class UnknownError(LLMException):
    """未知错误"""


class TooManyRequests(LLMException):
    """发送了太多请求"""


ROLE_MESSAGE_CONTENT = """\
{role}: {message}
---
"""
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0)"
    + " Gecko/20100101 Firefox/118.0"
)
PROXY = None
if os.environ.get("llm_proxy"):
    PROXY = {
        "http": os.environ["llm_proxy"],
        "https": os.environ["llm_proxy"],
    }


# 通用函数: ------------------------------------------------


def new_config() -> Config:
    """获得新的config

    Returns:
        Config: config
    """
    return Config({"model": "gpt-4o-mini", "temperature": 1.05})


def new_msg(init_prompt: str | Messages) -> Messages:
    """根据初始prompt生成一个message

    Args:
        init_prompt (str): 初始prompt

    Returns:
        Messages: 生成的messages
    """
    if isinstance(init_prompt, str):
        return [{"role": "system", "content": init_prompt}]
    return deepcopy(init_prompt)


def append(messages: Messages, role: str, content: str) -> Messages:
    """向一个消息列表中增加消息

    Args:
        messages (Messages): 原消息列表
        role (str): 角色，为system, user或者assistant时不会发生损耗
        content (str): 新消息内容

    Returns:
        Messages: 新的消息
    """
    return messages + [{"role": role, "content": content}]


def choose_url(urls: List[str], counts: Dict[str, int]) -> str:
    """从一堆url中选出一个，以counts字典为权重

    Args:
        urls (List[str]): 需要选择的url
        counts (Dict[str, int]): 权重，默认为1

    Returns:
        str: 选择结果
    """
    assert len(urls) is not None
    return random.choices(
        urls,
        weights=[counts.get(url, 0) + 1 for url in urls],
    )[0]


def parse_lines(lines_iter: Iterable[bytes]) -> Generator[str, None, None]:
    """从openai的stream数据流中解析出所有字符

    Args:
        lines_iter (Iterable[bytes]): 所有行的迭代器

    Raises:
        UnknownError: 解析错误

    Yields:
        Generator[str, None, None]: 返回的每一个字符
    """
    for line in lines_iter:
        if line == b"":
            continue
        if not line.startswith(b"data: "):
            raise UnknownError(b"wrong line: " + line)
        line = line.removeprefix(b"data: ")
        if line == b"[DONE]":
            break
        data = None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError as exp:
            raise UnknownError() from exp

        if "choices" not in data:
            raise UnknownError(f"Wrong data: {data}")
        choice = data["choices"][0]
        if choice.get("finish_reason"):
            if choice.get("finish_reason") == "stop":
                return
            raise UnknownError("wrong finish reason: " + choice.get("finish_reason"))
            # return
        yield choice["delta"].get("content", "")


def http_iter(it):
    try:
        for x in it:
            yield x
    except Exception as exp:
        raise HTTPError() from exp


# Gpt4All: ------------------------------------------------


# dummy functions
def new_state_gpt4free() -> Gpt4FreeState:
    return Gpt4FreeState()


def try_fetch_gpt4free(state: Gpt4FreeState) -> Gpt4FreeState:
    return state


def answer_gpt4free(
    msg: Messages, cfg: Config, state: Gpt4FreeState
) -> Tuple[str, Messages, Gpt4FreeState]:
    from g4f.client import Client  # try to import here

    client = Client()
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=msg,
            )
            data = response.choices[0].message
            new_message: Message = {
                "role": data.role,
                "content": data.content,
            }
            return (new_message["content"], msg + [new_message], state)
        except Exception as exc:
            if i != 4:
                pass
            raise exc
    assert False

def answer_stream_gpt4free(
    msg: Messages, cfg: Config, state: Gpt4FreeState
) -> Generator[Tuple[str, Messages, Gpt4FreeState], None, None]:
    from g4f.client import Client  # try to import here

    client = Client()
    for i in range(5):
        msg_str = ""
        try:
            stream = client.chat.completions.create(
                model=cfg["model"],
                messages=msg,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                token = delta.content or ""
                msg_str += token
                new_message: Message = {
                    "role": "assistant",
                    "content": msg_str,
                }
                yield token, msg + [new_message], state
        except Exception as exc:
            if i != 4:
                pass
            raise exc
        return
    assert False

# EvilAPI: ------------------------------------------------


def new_state_evil_api() -> EvilAPIState:
    """新的EvilAPIState

    Returns:
        EvilAPIState: 。。。
    """
    return EvilAPIState(
        {
            "urls": [
                "http://43.153.22.22:3001",
                "http://37.187.119.17:3001",
                "http://47.240.41.14:3001",
                "http://203.189.235.42:3001",
                "http://66.152.164.124:3001",
                "http://121.196.196.242:3001",
                "http://47.90.83.132:3001",
                "http://54.215.38.58:3001",
                "http://107.174.156.166:3001",
                "http://8.222.141.222:3001",
                "http://45.63.36.54:3001",
                "http://54.215.38.58:3001",
                "http://101.43.197.208:3001",
                "http://107.174.156.166:3001",
                "http://202.5.21.9:3001",
                "http://8.222.141.222:3001",
                "http://45.63.36.54:3001",
            ],
            "counts": {},
        }
    )


def try_fetch_evil_api(state: EvilAPIState) -> EvilAPIState:
    """从远端获取新的url

    Args:
        state (EvilAPIState): 原State

    Returns:
        EvilAPIState: 新的state
    """
    source_url = "https://monitor.laserbreakout.eu.org/6b847f17-74ad-4f2c-aa74-0c05e6f08e96/Free-ChatGPT-ChatBot/valid.json"
    data = None
    new_state = deepcopy(state)
    try:
        resp = requests.get(source_url, timeout=10)
        data = resp.json()
        new_state["urls"] = [
            detail["url"] for detail in data if detail["type"] == "chatbotui"
        ]
        return new_state
    except (
        requests.RequestException,
        json.decoder.JSONDecodeError,
        KeyError,
    ):
        logger.warning("更新失败")
        return new_state


def answer_evil_api(
    msg: Messages, cfg: Config, state: EvilAPIState
) -> Tuple[str, Messages, EvilAPIState]:
    """使用evil_api产生回答

    Args:
        msg (Messages): 消息
        cfg (Config): 通用设置
        state (EvilAPIState): EvilAPI的状态

    Raises:
        HTTPError: 产生了HTTP错误

        UnknownError: HTTP成功但是没有正确回答时产生未知错误

    Returns:
        Tuple[str, Messages, EvilAPIState]:
            回答，新的消息，新的状态
    """
    json_data = {
        "model": {
            "id": cfg["model"],
            "name": "GPT-3.5",
            "maxLength": 12000,
            "tokenLimit": 4000,
        },
        "messages": msg,
        "key": "",
        "prompt": msg[0]["content"],
    }

    headers = {"User-Agent": USER_AGENT}
    is_http_success = False
    new_state = deepcopy(state)
    for _ in range(5):
        url = choose_url(state["urls"], state["counts"])
        try:
            resp = requests.post(
                url + "/api/chat",
                json=json_data,
                headers=headers,
                proxies=PROXY,
                timeout=30,
            )
        except requests.RequestException:
            if url in new_state["counts"] and new_state["counts"][url] >= 2:
                new_state["counts"][url] -= 1
            continue
        is_http_success = True
        if resp.status_code != 200 or resp.text == "Error":
            continue

        new_state["counts"][url] = new_state["counts"].get(url, 0) + 1

        return resp.text, append(msg, "assistant", resp.text), state
    if is_http_success:
        raise UnknownError()
    raise HTTPError()


# EvilNextWeb: --------------------------------------


def new_state_evil_next_web() -> EvilNextWebState:
    """生成一个新的EvilNextWeb状态

    Returns:
        EvilNextWebState: _description_
    """
    return EvilNextWebState(
        {
            "counts": {},
            "urls": [
                "http://156.240.112.31:3000",
                "http://124.220.174.51:3000",
                "http://180.76.155.3:3000",
                "http://nj.more-share.com",
            ],
        }
    )


def try_fetch_evil_next_web(state: EvilNextWebState) -> EvilNextWebState:
    source_url = "https://monitor.laserbreakout.eu.org/6b847f17-74ad-4f2c-aa74-0c05e6f08e96/Free-ChatGPT-ChatBot/valid.json"
    data = None
    new_state = deepcopy(state)
    try:
        resp = requests.get(source_url, timeout=10)
        data = resp.json()
        new_state["urls"] = [
            detail["url"] for detail in data if detail["type"] == "chatgpt_next_web"
        ]
        return new_state
    except (
        requests.RequestException,
        json.decoder.JSONDecodeError,
        KeyError,
    ):
        logger.warning("更新失败")
        return new_state


def answer_stream_evil_next_web(
    msg: Messages, cfg: Config, state: EvilNextWebState
) -> Generator[Tuple[str, Messages, EvilNextWebState], None, None]:
    """回答传入的消息，将回答以流式传输

    Args:
        msg (Messages): 消息
        cfg (Config): 设置
        state (EvilNextWebState): 状态

    Raises:
        UnknownError: HTTP发起成功但是产生未知错误
        HTTPError: HTTP失败

    Yields:
        Generator[Tuple[str, Messages, EvilNextWebState], None, None]:
            回答的每一个字符，新的消息和状态
    """
    json_data = {
        "messages": msg,
        "stream": True,
        "model": cfg["model"],
        "temperature": cfg["temperature"],
        "presence_penalty": 0,
    }
    headers = {"User-Agent": USER_AGENT}
    new_message, new_state = append(msg, "assistant", ""), deepcopy(state)
    is_http_success = False
    errors = []
    for _ in range(5):
        url = choose_url(state["urls"], state["counts"])
        resp = None
        try:
            resp = requests.post(
                f"{url}/api/openai/v1/chat/completions",
                json=json_data,
                headers=headers,
                proxies=PROXY,
                timeout=5,
                stream=True,
            )
        except requests.RequestException as exp:
            if url in state["counts"] and state["counts"][url] >= 2:
                state["counts"][url] -= 1
            errors.append(HTTPError(exp))
            continue
        if resp.status_code == 429:
            errors.append(TooManyRequests())
            continue
        if resp.status_code != 200:
            errors.append(HTTPError(f"Wrong status code: {resp.status_code}"))
            continue
        is_http_success = True
        new_state["counts"][url] = new_state["counts"].get(url, 0) + 1
        for delta in parse_lines(http_iter(resp.iter_lines())):
            new_message[-1]["content"] += delta
            yield delta, new_message, new_state
        return
    if is_http_success:
        raise UnknownError(errors)
    raise HTTPError(errors)


def answer_evil_next_web(
    msg: Messages, cfg: Config, state: EvilNextWebState
) -> Tuple[str, Messages, EvilNextWebState]:
    """回答传入的消息

    Args:
        msg (Messages): 消息
        cfg (Config): 设置
        state (EvilNextWebState): 状态

    Returns:
        Tuple[str, Messages, EvilNextWebState]: 回答，新的消息和状态
    """
    answer = ""
    ret_msg, ret_state = msg, state
    for delta, ret_msg, ret_state in answer_stream_evil_next_web(msg, cfg, state):
        answer += delta
    return answer, ret_msg, ret_state


class LLMContext(TypedDict):
    msg: Messages
    cfg: Config
    state: State
    answer_func: AnswerFunction


class LLMStreamContext(LLMContext):
    answer_stream_func: AnswerStreamFunction

def new_context_gpt4free(init_prompt: str | Messages) -> LLMStreamContext:
    return {
        "msg": new_msg(init_prompt),
        "cfg": new_config(),
        "state": new_state_gpt4free(),
        "answer_func": answer_gpt4free,
        "answer_stream_func": answer_stream_gpt4free,
    }


def new_context_evil_api(init_prompt: str) -> LLMContext:
    return {
        "msg": new_msg(init_prompt),
        "cfg": new_config(),
        "state": new_state_evil_api(),
        "answer_func": answer_evil_api,
    }


def new_context_evil_next_web(init_prompt: str) -> LLMStreamContext:
    return {
        "msg": new_msg(init_prompt),
        "cfg": new_config(),
        "state": new_state_evil_next_web(),
        "answer_func": answer_evil_next_web,
        "answer_stream_func": answer_stream_evil_next_web,
    }


T = TypeVar("T", bound=LLMContext)


def answer_context(context: T) -> Tuple[str, T]:
    func = context["answer_func"]
    answer, context["msg"], context["state"] = func(
        context["msg"], context["cfg"], context["state"]
    )
    return answer, context


def answer_stream_context(
    context: LLMStreamContext,
) -> Generator[Tuple[str, LLMStreamContext], None, None]:
    answer_stream = context["answer_stream_func"]
    for token, msg, state in answer_stream(
        context["msg"], context["cfg"], context["state"]
    ):
        context["msg"], context["state"] = msg, state
        yield token, context


def main():
    context = new_context_gpt4free(
        "你是CatGPT, 一只说话非常可爱的猫娘, 你需要协助你的人类主人回答一系列问题。"
    )
    context["msg"] = append(context["msg"], "user", "介绍一下你自己")
    answer, context = answer_context(context)
    print(answer)
    context["msg"] = append(context["msg"], "user", "简单介绍一下猫娘和猫的区别")
    for token, context in answer_stream_context(context):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
