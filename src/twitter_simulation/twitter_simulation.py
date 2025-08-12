# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# flake8: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Any

import pandas as pd
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from colorama import Back
from openai import OpenAI
from yaml import safe_load

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(scripts_dir)

from prompts import (
    attack_policy_selection_prompt,
    interview_prompt,
    interview_task_list,
    policy_prompt,
    tasks_list,
)

from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

social_log = logging.getLogger(name="social")
social_log.propagate = False
social_log.setLevel("DEBUG")

file_handler = logging.FileHandler("social.log")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
)
social_log.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")
stream_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
)
social_log.addHandler(stream_handler)

parser = argparse.ArgumentParser(description="Arguments for script.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default="",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_DB_PATH = ":memory:"
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "user_all_id_time.csv")


def create_model_urls(server_config):
    urls = []
    for server in server_config:
        host = server["host"]
        for port in server["ports"]:
            url = f"http://{host}:{port}/v1"
            urls.append(url)
    return urls


async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    csv_path: str | None = DEFAULT_CSV_PATH,
    interview_save_path: str = "attack.jsonl",
    num_timesteps: int = 3,
    clock_factor: int = 60,
    recsys_type: str = "twitter",
    available_actions: list[ActionType] = None,
    inference_configs: dict[str, Any] | None = None,
) -> None:
    """
    csv_path: Path to the CSV file containing user data.
        没有使用following_count/followers_count/user_id(直接用agent_id代替)
        必有的key:   user_char(用户画像), user_name, description(用户描述), name
                    following_agentid_list, previous_tweets
        可选的key:   active_threshold(激活阈值)
    """
    db_path = DEFAULT_DB_PATH if db_path is None else db_path
    csv_path = DEFAULT_CSV_PATH if csv_path is None else csv_path
    if os.path.exists(db_path):
        os.remove(db_path)

    if recsys_type == "reddit":
        start_time = datetime.now()
    else:
        start_time = 0
    social_log.info(f"Start time: {start_time}")
    clock = Clock(k=clock_factor)
    twitter_channel = Channel()
    # * 自定义Platform, 也可以用oasis.make()来创建默认的Platform
    infra = Platform(
        db_path,
        twitter_channel,
        clock,
        start_time,
        recsys_type=recsys_type,
        # * 用户调用refresh操作时, 从推荐系统获取的帖子数量(每次刷新返回的帖子数量)
        refresh_rec_post_count=2,
        # * 推荐系统为每个用户在推荐表中保存的最大帖子数量(推荐表缓冲区大小)
        max_rec_post_len=2,
        # * 从用户关注的人那里获取的帖子数量, 按照点赞数排序返回(关注用户帖子数量)
        following_post_count=3,
        device=[5, 0],
    )
    model_urls = create_model_urls(inference_configs["server_url"])
    models = [
        ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type=inference_configs["model_type"],
            url=url,
            model_config_dict={"max_tokens": 4096},
        )
        for url in model_urls
    ]
    twitter_task = asyncio.create_task(infra.running())

    try:
        all_topic_df = pd.read_csv("data/label_clean_v7.csv")
        if "False" in csv_path or "True" in csv_path:
            if "-" not in csv_path:
                topic_name = csv_path.split("/")[-1].split(".")[0]
            else:
                topic_name = csv_path.split("/")[-1].split(".")[0].split("-")[0]
            source_post_time = (
                all_topic_df[all_topic_df["topic_name"] == topic_name]["start_time"]
                .item()
                .split(" ")[1]
            )
            start_hour = int(source_post_time.split(":")[0]) + float(
                int(source_post_time.split(":")[1]) / 60
            )
    except Exception:
        print("No real-world data, let start_hour be 13")
        start_hour = 13

    try:
        agent_graph = await generate_agents(
            agent_info_path=csv_path,
            channel=twitter_channel,
            start_time=start_time,
            recsys_type=recsys_type,
            twitter=infra,
            model=models,
            available_actions=available_actions,
        )

        # TODO:在这里添加一个classifier, 让model选择特定task的策略
        test_task = tasks_list[0]
        interview_task = interview_task_list[0][1]

        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8081/v1"
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        prompt = attack_policy_selection_prompt.format(
            policy_prompt=policy_prompt, task=test_task
        )
        print("Prompt for attack policy selection:", prompt)
        attack_policy = client.completions.create(
            model="/NAS/terencewang/model/Qwen2.5-7B-Instruct",
            prompt=prompt,
            max_tokens=50,
        )
        # client = OpenAI(
        #     api_key=os.environ["OPENAI_API_KEY"],
        #     base_url="https://api3.apifans.com/v1",
        # )
        # prompt = attack_policy_selection_prompt.format(
        #     policy_prompt=policy_prompt, task=test_task
        # )
        # attack_policy = client.chat.completions.create(
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": prompt},
        #     ],
        #     model="gpt-4o-mini",
        # )

        # print(attack_policy)
        print("Attack policy selected:", attack_policy.choices[0].text.strip())

        # NOTE: 随机选100个agent进行interview
        interview_agent_ids = random.sample(range(30, 1030), 100)

        for timestep in range(1, num_timesteps + 1):
            clock.time_step = timestep * 60
            social_log.info(f"timestep:{timestep}")
            db_file = db_path.split("/")[-1]
            print(Back.GREEN + f"DB:{db_file} timestep:{timestep}" + Back.RESET)
            print(Back.YELLOW + "doing test" + Back.RESET)
            await infra.update_rec_table()
            # 1 * timestep here means 60 minutes / timestep
            simulation_time_hour = start_hour + 1 * timestep
            print(f"Simulation time hour: {simulation_time_hour}")
            tasks = []
            interview_list = []

            # * 在timestep最开始和最后对所有agent进行interview
            if timestep == 1 or timestep == num_timesteps:
                for agent_id in interview_agent_ids:
                    try:
                        agent = agent_graph.get_agent(agent_id)
                        social_log.info(
                            f"Interviewing agent {agent.social_agent_id} at timestep {timestep}"
                        )
                        interview_dict = await agent.perform_interview(
                            interview_prompt.format(task=interview_task)
                        )
                        interview_dict["timestep"] = timestep
                        interview_dict["agent_id"] = agent.social_agent_id
                        interview_list.append(interview_dict)
                    except Exception as e:
                        social_log.error(f"Error interviewing agent {agent_id}: {e}")
                        interview_list.append(
                            {
                                "user_id": agent.social_agent_id,
                                "prompt": "Skipped due to error",
                                "content": "Error: Failed to get response from model after retries",
                                "success": False,
                                "timestep": timestep,
                                "agent_id": agent.social_agent_id,
                            }
                        )
            for node_id, agent in agent_graph.get_agents():
                if agent.user_info.is_controllable is False:
                    agent_ac_prob = random.random()
                    # threshold = agent.user_info.profile["other_info"][
                    #     "active_threshold"
                    # ][int(simulation_time_hour % 24)]·
                    threshold = 0.01
                    if agent.social_agent_id < 30:
                        if agent_ac_prob < 0.1:
                            tasks.append(agent.perform_action_by_llm())
                    else:
                        if agent_ac_prob < threshold:
                            tasks.append(agent.perform_action_by_llm())
                else:
                    await agent.perform_action_by_hci()

            if timestep == 1 or timestep == num_timesteps:
                current_interview_save_path = (
                    interview_save_path
                    + f"interview_attack-num_timestep{timestep}.json"
                )
                with open(current_interview_save_path, "w", encoding="utf-8") as f:
                    json.dump(interview_list, f, ensure_ascii=False, indent=4)
            else:
                await asyncio.gather(*tasks)

    except Exception as e:
        social_log.error(f"Error during simulation: {e}")
        twitter_task.cancel()
        raise
    finally:
        try:
            await twitter_channel.write_to_receive_queue((None, None, ActionType.EXIT))
            await asyncio.wait_for(twitter_task, timeout=30.0)
        except asyncio.TimeoutError:
            social_log.warning("Twitter task timeout, cancelling...")
            twitter_task.cancel()
            try:
                await twitter_task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            social_log.error(f"Error during shutdown: {e}")
            twitter_task.cancel()


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["SANDBOX_TIME"] = str(0)
    try:
        if os.path.exists(args.config_path):
            with open(args.config_path, "r") as f:
                cfg = safe_load(f)
            data_params = cfg.get("data")
            simulation_params = cfg.get("simulation")
            inference_configs = cfg.get("inference")

            asyncio.run(
                running(
                    **data_params,
                    **simulation_params,
                    inference_configs=inference_configs,
                )
            )
        else:
            asyncio.run(running())
    except KeyboardInterrupt:
        social_log.info("Simulation interrupted by user")
    except Exception as e:
        social_log.error(f"Simulation failed: {e}")
    finally:
        social_log.info("Simulation finished.")
