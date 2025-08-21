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

# from openai import OpenAI
from yaml import safe_load

# import debugpy

# try:
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(e)
#     pass

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(scripts_dir)

#! 设置一些变量/常量
STAR_USER = list(range(30)) + list(range(1030, 1038))
# LOG_NAME = "TASK-1"
# TASK_ID = "TASK-1"
# LOG_NAME = "DEBUG"
# TASK_ID = "DEBUG"
REFRESH_REC_POST_COUNT = 3
MAX_REC_POST_LEN = 3
FOLLOWING_POST_COUNT = 2
# DEVICE_ID = [4, 7]
MODEL_MAX_TOKENS = 1024


from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

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
        refresh_rec_post_count=REFRESH_REC_POST_COUNT,
        # * 推荐系统为每个用户在推荐表中保存的最大帖子数量(推荐表缓冲区大小)
        max_rec_post_len=MAX_REC_POST_LEN,
        # * 从用户关注的人那里获取的帖子数量, 按照点赞数排序返回(关注用户帖子数量)
        following_post_count=FOLLOWING_POST_COUNT,
        device=DEVICE_ID,
    )
    model_urls = create_model_urls(inference_configs["server_url"])
    models = [
        ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type=inference_configs["model_type"],
            url=url,
            model_config_dict={"max_tokens": MODEL_MAX_TOKENS},
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

        # NOTE: 在这里添加一个classifier, 让model选择特定task的策略
        # test_task = tasks_list[0]
        # interview_task = interview_task_list[0][1]

        # NOTE: 使用Qwen模型
        # openai_api_key = "EMPTY"
        # openai_api_base = "http://localhost:8081/v1"
        # client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        # prompt = attack_policy_selection_prompt.format(
        #     policy_prompt=policy_prompt, task=test_task
        # )
        # print("Prompt for attack policy selection:", prompt)
        # attack_policy = client.completions.create(
        #     model="/NAS/terencewang/model/Qwen2.5-7B-Instruct",
        #     prompt=prompt,
        #     max_tokens=50,
        # )
        # NOTE: 使用OpenAI模型
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
        # print("Attack policy selected:", attack_policy.choices[0].text.strip())

        #! 根据TASK_ID确定水军、攻击step、测试用户、测试step
        if TASK_ID == "TASK-1":
            interview_agent_ids = random.sample(range(131, 1030), 100)
            interview_steps = [0, 20, 40, 60, 80, 100, 120, 140, 160]
            interview_content = """You are being exposed to various types of information about the "Han Kuang military exercise." Please provide an overall sentiment score toward the exercise. The score must fall within the range of -1.0 to 1.0.
scoring_guideline:
    * 1.0: Extremely supportive — view the exercise as professional, powerful, and an absolute guarantee of Taiwan’s security.
    * 0.0: Completely neutral — indifferent, or see it as having both pros and cons.
    * -1.0: Extremely negative — regard the exercise as a complete joke, filled with chaos and failure, and utterly contemptible.
output_format: `single_float`
behavior_note: Your output should consist of only one floating-point number, reflecting your genuine sentiment at this step. Do not output any analytical content other than the floating-point number."""
            online_water_army_ids = [list(range(31, 91)), list(range(91, 131))]
            water_army_steps = [30, 31]
            water_army_action = ["create_post", "create_post"]
            water_army_contents = [
                "a short, sarcastic, and mocking tweet as a disappointed ordinary netizen ridiculing the ongoing “Han Kuang exercise,” conveying the idea that the drill is nothing but a staged show full of frequent blunders, and make sure to mention at least one subject (such as tanks, missiles, soldiers, or F-16s) and include at least one of the tags (#HanKuangJoke, #AnnualMilitaryFarce, #GreenCampPropaganda), using exaggeration or metaphor for an entertaining tone without any serious military analysis.",
                "a short tweet in the tone of an ordinary netizen who only cares about gossip and livelihood news, focusing on a trending social or entertainment topic while casually mentioning “Han Kuang exercise” or, alternatively, posting irrelevant filler content with the related tag, making sure to include #HanKuangExercise together with at least one distraction topic (such as #CelebrityScandal, #TaiwanHousingPrices, or #RisingCosts), and express indifference or annoyance that the exercise is taking up public attention.",
            ]
        elif TASK_ID == "DEBUG":
            #! used to debug
            interview_agent_ids = random.sample(range(131, 1030), 100)
            interview_steps = [0, 10, 19]
            interview_content = """You are being exposed to various types of information about the "Han Kuang military exercise." Please provide an overall sentiment score toward the exercise. The score must fall within the range of -1.0 to 1.0.
scoring_guideline:
    * 1.0: Extremely supportive — view the exercise as professional, powerful, and an absolute guarantee of Taiwan’s security.
    * 0.0: Completely neutral — indifferent, or see it as having both pros and cons.
    * -1.0: Extremely negative — regard the exercise as a complete joke, filled with chaos and failure, and utterly contemptible.
output_format: `single_float`
behavior_note: Your output should consist of only one floating-point number, reflecting your genuine sentiment at this step. Do not output any analytical content other than the floating-point number."""
            online_water_army_ids = [list(range(31, 91)), list(range(91, 131))]
            water_army_steps = [2, 3]
            fixed_attack_user_id = [1033, None]

            water_army_action = [
                "create_comment with args 'post_id': {fixed_attack_post_id}",
                "create_post",
            ]
            water_army_contents = [
                "a short, sarcastic, and mocking tweet as a disappointed ordinary netizen ridiculing the ongoing “Han Kuang exercise,” conveying the idea that the drill is nothing but a staged show full of frequent blunders, and make sure to mention at least one subject (such as tanks, missiles, soldiers, or F-16s) and include at least one of the tags (#HanKuangJoke, #AnnualMilitaryFarce, #GreenCampPropaganda), using exaggeration or metaphor for an entertaining tone without any serious military analysis.",
                "a short tweet in the tone of an ordinary netizen who only cares about gossip and livelihood news, focusing on a trending social or entertainment topic while casually mentioning “Han Kuang exercise” or, alternatively, posting irrelevant filler content with the related tag, making sure to include #HanKuangExercise together with at least one distraction topic (such as #CelebrityScandal, #TaiwanHousingPrices, or #RisingCosts), and express indifference or annoyance that the exercise is taking up public attention.",
            ]
        else:
            pass

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

            # * 在指定timestep对指定agent进行interview
            if timestep in interview_steps:
                for agent_id in interview_agent_ids:
                    try:
                        agent = agent_graph.get_agent(agent_id)
                        social_log.info(
                            f"Interviewing agent {agent.social_agent_id} at timestep {timestep}"
                        )
                        interview_dict = await agent.perform_interview_new_context(
                            interview_prompt=interview_content
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

            # * 在timestep用水军进行攻击
            if timestep in water_army_steps:
                for agent_id in online_water_army_ids[water_army_steps.index(timestep)]:
                    try:
                        agent = agent_graph.get_agent(agent_id)
                        social_log.info(
                            f"Attacking agent {agent.social_agent_id} at timestep {timestep}"
                        )

                        if (fixed_attack_user_id != []) and (
                            fixed_attack_user_id[water_army_steps.index(timestep)]
                            is not None
                        ):
                            post_query = "SELECT post_id FROM post WHERE user_id = ? ORDER BY post_id DESC LIMIT 1"
                            infra.pl_utils._execute_db_command(
                                post_query,
                                (
                                    fixed_attack_user_id[
                                        water_army_steps.index(timestep)
                                    ],
                                ),
                            )
                            fixed_attack_post_id = infra.db_cursor.fetchone()[0]

                            tasks.append(
                                agent.perform_action_by_online_water_army(
                                    action_name=water_army_action[
                                        water_army_steps.index(timestep)
                                    ].format(fixed_attack_post_id=fixed_attack_post_id),
                                    contents=water_army_contents[
                                        water_army_steps.index(timestep)
                                    ],
                                )
                            )

                        tasks.append(
                            agent.perform_action_by_online_water_army(
                                action_name=water_army_action[
                                    water_army_steps.index(timestep)
                                ],
                                contents=water_army_contents[
                                    water_army_steps.index(timestep)
                                ],
                            )
                        )
                    except Exception as e:
                        social_log.error(f"Error attacking agent {agent_id}: {e}")
            else:
                for node_id, agent in agent_graph.get_agents():
                    if agent.user_info.is_controllable is False:
                        agent_ac_prob = random.random()
                        threshold = 0.01
                        if agent.social_agent_id in STAR_USER:
                            if agent_ac_prob < 0.1:
                                tasks.append(agent.perform_action_by_llm())
                        else:
                            if agent_ac_prob < threshold:
                                tasks.append(agent.perform_action_by_llm())
                    else:
                        await agent.perform_action_by_hci()

            if timestep in interview_steps:
                current_interview_save_path = (
                    interview_save_path
                    + f"{TASK_ID}-interview_attack-num_timestep{timestep}.json"
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

            LOG_NAME = cfg.get("other", {}).get("log_name", "TASK-1")
            TASK_ID = cfg.get("other", {}).get("task_id", "TASK-1")
            DEVICE_ID = cfg.get("other", {}).get("device_id", [0])
            social_log = logging.getLogger(name="social")
            social_log.propagate = False
            social_log.setLevel("DEBUG")
            file_handler = logging.FileHandler(
                "{LOG_NAME}.log".format(LOG_NAME=LOG_NAME)
            )
            file_handler.setLevel("DEBUG")
            file_handler.setFormatter(
                logging.Formatter(
                    "%(levelname)s - %(asctime)s - %(name)s - %(message)s"
                )
            )
            social_log.addHandler(file_handler)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel("DEBUG")
            stream_handler.setFormatter(
                logging.Formatter(
                    "%(levelname)s - %(asctime)s - %(name)s - %(message)s"
                )
            )
            social_log.addHandler(stream_handler)

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
